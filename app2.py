# app2.py
# 逐词选择（基于前缀树）来构造条件；每次变更立即生成图像
import os
import math
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
from torch import nn
import numpy as np
from PIL import Image, ImageTk

# 从你之前的文件导入
from build_caption_trie import CaptionTrie
try:
    # 可选：若你在 build_caption_trie.py 中也定义了 build_trie_from_csv
    from build_caption_trie import build_trie_from_csv
except Exception:
    build_trie_from_csv = None


# --------------------------
# 与 train_conv.py 对齐的模型定义
# --------------------------
def conv_block(in_ch, out_ch, downsample=True):
    stride = 2 if downsample else 1
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )

def dec_block(in_ch, out_ch, upsample=True):
    layers = []
    if upsample:
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
    layers += [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    return nn.Sequential(*layers)

class CVAEConv(nn.Module):
    def __init__(self, img_channels, img_size, cond_dim, latent_dim=64, base_ch=32):
        super().__init__()
        self.img_channels = img_channels
        self.H, self.W = img_size
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        curr_ch = img_channels
        curr_h, curr_w = self.H, self.W
        self.enc_blocks = nn.ModuleList()
        plan = []
        for out_ch in [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]:
            if min(curr_h // 2, curr_w // 2) < 2:
                self.enc_blocks.append(conv_block(curr_ch, out_ch, downsample=False))
                curr_ch = out_ch
                plan.append((out_ch, False))
                break
            self.enc_blocks.append(conv_block(curr_ch, out_ch, downsample=True))
            curr_ch = out_ch
            curr_h //= 2
            curr_w //= 2
            plan.append((out_ch, True))

        self.enc_out_ch = curr_ch
        self.enc_out_h = curr_h
        self.enc_out_w = curr_w
        self.enc_feat_dim = self.enc_out_ch * self.enc_out_h * self.enc_out_w

        self.fc_enc = nn.Sequential(
            nn.Linear(self.enc_feat_dim + cond_dim, max(256, latent_dim * 4)),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(max(256, latent_dim * 4), latent_dim)
        self.logvar_head = nn.Linear(max(256, latent_dim * 4), latent_dim)

        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, max(256, latent_dim * 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(256, latent_dim * 4), self.enc_feat_dim),
            nn.ReLU(inplace=True),
        )

        dec_blocks = []
        dec_ch = self.enc_out_ch
        for out_ch, down in reversed(plan):
            dec_blocks.append(dec_block(dec_ch, out_ch, upsample=down))
            dec_ch = out_ch
        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.dec_head = nn.Sequential(
            nn.Conv2d(dec_ch, img_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x, c):
        h = x
        for blk in self.enc_blocks:
            h = blk(h)
        h = h.view(h.size(0), -1)
        hc = torch.cat([h, c], dim=-1)
        hc = self.fc_enc(hc)
        mu = self.mu_head(hc)
        logvar = self.logvar_head(hc)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        h = self.fc_dec(zc)
        h = h.view(h.size(0), self.enc_out_ch, self.enc_out_h, self.enc_out_w)
        for blk in self.dec_blocks:
            h = blk(h)
        x_hat = self.dec_head(h)
        return x_hat


def infer_hparams_from_state_dict(state_dict):
    base_ch = int(state_dict["enc_blocks.0.0.weight"].shape[0])
    latent_dim = int(state_dict["mu_head.bias"].shape[0])
    return base_ch, latent_dim


# --------------------------
# Tk App：基于 Trie 的逐词选择
# --------------------------
class TrieSelectApp(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.title("Pixel CVAE - Trie Word Picker")
        self.args = args

        # 设备
        self.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        torch.manual_seed(args.seed)

        # 加载/构建 Trie
        if args.trie and os.path.exists(args.trie):
            self.trie = CaptionTrie.load(args.trie)
        elif args.csv and build_trie_from_csv is not None:
            self.trie = build_trie_from_csv(args.csv)
            if args.trie:
                try:
                    self.trie.save(args.trie)
                except Exception as e:
                    print(f"[Warn] 保存Trie失败: {e}")
        else:
            messagebox.showerror("Error", "未找到有效的 Trie。请提供 --trie 或 --csv 以构建。")
            raise SystemExit(1)

        # 读取 vocab（token->idx）
        if not args.vocab or (not os.path.exists(args.vocab)):
            messagebox.showwarning("Warning", "未提供 vocab.json，将无法将选择映射为条件向量（使用全0）。")
            self.vocab = {}
        else:
            import json
            with open(args.vocab, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)

        # 加载模型 ckpt
        ckpt = torch.load(args.ckpt, map_location=self.device)
        self.state_dict = ckpt["model_state_dict"]
        if "img_shape" not in ckpt:
            messagebox.showerror("Error", "Checkpoint 缺少 'img_shape'。请用 train_conv.py 重新训练生成。")
            raise SystemExit(1)
        self.C, self.H, self.W = ckpt["img_shape"]
        self.cond_dim_ckpt = ckpt.get("cond_dim", None)
        self.cond_dim = args.cond_dim if args.cond_dim is not None else (self.cond_dim_ckpt or 270)
        if self.cond_dim_ckpt is not None and self.cond_dim != self.cond_dim_ckpt:
            print(f"[Warn] cond_dim mismatch: ckpt={self.cond_dim_ckpt}, cli={self.cond_dim}. "
                  f"使用 cli 值 {self.cond_dim}。")

        base_ch, latent_dim = infer_hparams_from_state_dict(self.state_dict)
        self.latent_dim = latent_dim
        self.model = CVAEConv(
            img_channels=self.C,
            img_size=(self.H, self.W),
            cond_dim=self.cond_dim,
            latent_dim=self.latent_dim,
            base_ch=base_ch,
        ).to(self.device)
        self.model.load_state_dict(self.state_dict, strict=True)
        self.model.eval()

        # 交互状态
        self.z = torch.randn(1, self.latent_dim, device=self.device)
        self.prefix_tokens = []  # 已选择的前缀
        self.c = torch.zeros(1, self.cond_dim, device=self.device)

        # UI
        self._build_ui()

        # 初始化建议+图像
        self._refresh_suggestions()
        self._schedule_infer()

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        # 顶部：图像与工具
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.img_label = ttk.Label(top, text="(生成图在此显示)")
        self.img_label.pack(side=tk.LEFT, padx=8)

        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self.status_var).pack(anchor="w", pady=(0, 6))

        row1 = ttk.Frame(right)
        row1.pack(anchor="w", pady=2)
        ttk.Button(row1, text="Resample z", command=self._resample_z).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Save PNG", command=self._save_png).pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(right)
        row2.pack(anchor="w", pady=2)
        ttk.Button(row2, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=2)

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=8, pady=(4, 6))

        # 中部：前缀显示
        prefix_frame = ttk.Frame(self)
        prefix_frame.pack(side=tk.TOP, fill=tk.X, padx=8)
        ttk.Label(prefix_frame, text="Prefix:").pack(side=tk.LEFT)
        self.prefix_var = tk.StringVar(value="")
        self.prefix_label = ttk.Label(prefix_frame, textvariable=self.prefix_var)
        self.prefix_label.pack(side=tk.LEFT, padx=6)

        # 底部：候选词 Listbox（可滚动） + “添加”按钮
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(6, 8))

        left = ttk.Frame(bottom)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Next tokens:").pack(anchor="w")

        self.listbox = tk.Listbox(left, height=14, exportselection=False)
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=yscroll.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 双击选择
        self.listbox.bind("<Double-Button-1>", lambda e: self._add_selected())

        controls = ttk.Frame(bottom)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(controls, text="Add ➜", command=self._add_selected).pack(pady=4, fill=tk.X)

        # 可直接输入一个 token 添加
        ttk.Label(controls, text="Manual token:").pack(anchor="w", pady=(8, 2))
        self.manual_entry = ttk.Entry(controls)
        self.manual_entry.pack(fill=tk.X)
        self.manual_entry.bind("<Return>", lambda e: self._add_manual())

        ttk.Button(controls, text="Add manual", command=self._add_manual).pack(pady=4, fill=tk.X)

    # ------------- 逻辑 -------------
    def _refresh_prefix_view(self):
        self.prefix_var.set(" ".join(self.prefix_tokens))

    def _refresh_suggestions(self):
        # 基于前缀从 Trie 获取下一层候选
        try:
            candidates = self.trie.next_tokens(self.prefix_tokens)
        except Exception:
            candidates = []
        # 限制 topk（按字母序）
        candidates = sorted(candidates)
        if self.args.topk > 0:
            candidates = candidates[: self.args.topk]

        self.listbox.delete(0, tk.END)
        for tok in candidates:
            self.listbox.insert(tk.END, tok)

    def _add_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        tok = self.listbox.get(sel[0])
        self._push_token(tok)

    def _add_manual(self):
        tok = self.manual_entry.get().strip().lower()
        tok = "".join([ch for ch in tok if (ch.isalpha() or ch == " ")])  # 简单清洗
        tok = tok.strip()
        if not tok:
            return
        self._push_token(tok)
        self.manual_entry.delete(0, tk.END)

    def _push_token(self, tok: str):
        # 尝试通过 Trie 走一步；若当前前缀下不存在该分支，也允许加入，但后续候选会为空
        # 这样可以自由编辑，但会提示
        valid = tok in (self.trie.next_tokens(self.prefix_tokens) or [])
        if not valid:
            print(f"[Warn] '{tok}' 不在当前前缀的候选列表中。")
        self.prefix_tokens.append(tok)
        self._refresh_prefix_view()
        self._update_condition_from_prefix()
        self._refresh_suggestions()
        self._schedule_infer()

    def _undo(self):
        if self.prefix_tokens:
            self.prefix_tokens.pop()
            self._refresh_prefix_view()
            self._update_condition_from_prefix()
            self._refresh_suggestions()
            self._schedule_infer()

    def _clear(self):
        self.prefix_tokens = []
        self._refresh_prefix_view()
        self._update_condition_from_prefix()
        self._refresh_suggestions()
        self._schedule_infer()

    # 将前缀映射为 BOW 条件向量
    def _update_condition_from_prefix(self):
        self.c.zero_()
        if not self.vocab:
            # 没有 vocab，保持全0
            return
        for tok in self.prefix_tokens:
            idx = self.vocab.get(tok, None)
            if idx is not None and 0 <= idx < self.cond_dim:
                self.c[0, idx] = 1.0
            else:
                # 忽略未在 vocab 的词
                pass

    def _schedule_infer(self, delay_ms=50):
        if hasattr(self, "_infer_job") and self._infer_job is not None:
            self.after_cancel(self._infer_job)
        self._infer_job = self.after(delay_ms, self._infer_and_update)

    def _infer_and_update(self):
        self._infer_job = None
        try:
            self.status_var.set("Generating...")
            self.update_idletasks()
            with torch.no_grad():
                imgs = self.model.decode(self.z, self.c.to(self.device))
            img = imgs[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            pil = Image.fromarray((img * 255).astype(np.uint8))
            pil_small = pil.resize((128, 128), resample=Image.NEAREST)
            self.tk_img = ImageTk.PhotoImage(pil_small)
            self.img_label.configure(image=self.tk_img)
            self.status_var.set("Done")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            print("Inference error:", e)

    def _resample_z(self):
        self.z = torch.randn(1, self.latent_dim, device=self.device)
        self._schedule_infer()

    def _save_png(self):
        try:
            file = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")],
                initialfile="cvae_trie_sample.png",
            )
            if not file:
                return
            with torch.no_grad():
                imgs = self.model.decode(self.z, self.c.to(self.device))
            img = imgs[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            pil = Image.fromarray((img * 255).astype(np.uint8))
            pil_small = pil.resize((128, 128), resample=Image.NEAREST)
            pil_small.save(file)
            messagebox.showinfo("Saved", f"Saved to {file}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# --------------------------
# 入口
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="./models_conv/last.pt", help="训练得到的checkpoint路径")
    p.add_argument("--trie", type=str, default="./caption_trie.json", help="前缀树JSON；若不存在且提供--csv则动态构建")
    p.add_argument("--csv", type=str, default=None, help="可选：CSV路径（image, caption），若--trie不存在则用它构建")
    p.add_argument("--vocab", type=str, default="./vocab.json", help="token->idx 映射，用于条件BOW")
    p.add_argument("--cond_dim", type=int, default=270, help="条件向量维度（默认270）")
    p.add_argument("--topk", type=int, default=64, help="候选展示的最大数量（按字母序截断，0=不限）")
    p.add_argument("--cpu", action="store_true", help="强制用CPU")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    app = TrieSelectApp(args)
    app.mainloop()
