# app2.py — 可配置 UI 的 Trie 逐词选择器
# 通过 UIConfig 统一配置 选框/字体/图像大小 等
import os
import math
import json
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

import torch
from torch import nn
import numpy as np
from PIL import Image, ImageTk

from build_caption_trie import CaptionTrie
try:
    from build_caption_trie import build_trie_from_csv
except Exception:
    build_trie_from_csv = None


# =========================
# 与 train_conv.py 对齐的模型
# =========================
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


# =========================
# UI 配置（抽离）
# =========================
_RESAMPLE_MAP = {
    "NEAREST": Image.NEAREST,
    "BILINEAR": Image.BILINEAR,
    "BICUBIC": Image.BICUBIC,
    "LANCZOS": Image.LANCZOS,
}

class UIConfig:
    """
    所有可视化参数集中管理：
      - 字体/字号
      - 控件间距
      - 列表高度/宽度
      - 图像显示尺寸/插值方法
      - 主题
    支持 JSON 文件加载 + CLI 覆盖
    """
    def __init__(self, d: dict | None = None):
        d = d or {}
        # Font
        self.font_family: str = d.get("font_family", "Arial")
        self.font_size_base: int = int(d.get("font_size_base", 11))
        self.font_size_title: int = int(d.get("font_size_title", 12))
        self.bold_title: bool = bool(d.get("bold_title", True))

        # Spacing / padding
        self.pad: int = int(d.get("pad", 8))
        self.inner_pad: int = int(d.get("inner_pad", 6))
        self.button_padx: int = int(d.get("button_padx", 2))
        self.button_pady: int = int(d.get("button_pady", 2))

        # Listbox
        self.listbox_height: int = int(d.get("listbox_height", 14))
        self.listbox_width: int = int(d.get("listbox_width", 32))

        # Image display
        self.image_width: int = int(d.get("image_width", 128))
        self.image_height: int = int(d.get("image_height", 128))
        self.resample: str = str(d.get("resample", "NEAREST")).upper()
        if self.resample not in _RESAMPLE_MAP:
            self.resample = "NEAREST"

        # Theme (ttk style theme，可根据系统安装情况选择)
        self.theme: str = d.get("theme", "clam")  # 'alt', 'default', 'clam', 'vista'...

    @classmethod
    def from_json(cls, path: str | None):
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return cls(d)
        return cls({})

    def override_with_args(self, args):
        # 仅覆盖传入的 CLI 选项（None 则不动）
        for k in [
            "font_family", "font_size_base", "font_size_title", "bold_title",
            "pad", "inner_pad", "button_padx", "button_pady",
            "listbox_height", "listbox_width",
            "image_width", "image_height", "resample", "theme",
        ]:
            v = getattr(args, k, None)
            if v is not None:
                if k in {"font_size_base", "font_size_title", "pad", "inner_pad",
                         "button_padx", "button_pady", "listbox_height", "listbox_width",
                         "image_width", "image_height"}:
                    v = int(v)
                if k == "bold_title":
                    v = bool(v)
                if k == "resample":
                    v = str(v).upper()
                    if v not in _RESAMPLE_MAP:  # 防错
                        v = self.resample
                setattr(self, k, v)

    # 便捷：获取 PIL 插值常量
    @property
    def pil_resample(self):
        return _RESAMPLE_MAP[self.resample]


# =========================
# Tk App：基于 Trie 的逐词选择（使用 UIConfig）
# =========================
class TrieSelectApp(tk.Tk):
    def __init__(self, args, ui: UIConfig):
        super().__init__()
        self.title("Pixel CVAE - Trie Word Picker (Configurable UI)")
        self.args = args
        self.ui = ui

        # 设备
        self.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        torch.manual_seed(args.seed)

        # 主题 & 字体
        style = ttk.Style(self)
        try:
            style.theme_use(self.ui.theme)
        except Exception:
            pass  # 不支持的主题则忽略

        self.font_base = tkfont.Font(family=self.ui.font_family, size=self.ui.font_size_base)
        self.font_title = tkfont.Font(
            family=self.ui.font_family,
            size=self.ui.font_size_title,
            weight="bold" if self.ui.bold_title else "normal",
        )

        # 设置 ttk 默认字体
        style.configure(".", font=self.font_base)
        style.configure("Title.TLabel", font=self.font_title)

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

        # 读取 vocab
        if not args.vocab or (not os.path.exists(args.vocab)):
            messagebox.showwarning("Warning", "未提供 vocab.json，将使用全0条件向量。")
            self.vocab = {}
        else:
            with open(args.vocab, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)

        # 模型
        ckpt = torch.load(args.ckpt, map_location=self.device)
        self.state_dict = ckpt["model_state_dict"]
        if "img_shape" not in ckpt:
            messagebox.showerror("Error", "Checkpoint 缺少 'img_shape'。请用 train_conv.py 重新训练生成。")
            raise SystemExit(1)
        self.C, self.H, self.W = ckpt["img_shape"]
        self.cond_dim_ckpt = ckpt.get("cond_dim", None)
        self.cond_dim = args.cond_dim if args.cond_dim is not None else (self.cond_dim_ckpt or 270)
        if self.cond_dim_ckpt is not None and self.cond_dim != self.cond_dim_ckpt:
            print(f"[Warn] cond_dim mismatch: ckpt={self.cond_dim_ckpt}, cli={self.cond_dim}，使用 cli 值。")

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
        self.prefix_tokens = []
        self.c = torch.zeros(1, self.cond_dim, device=self.device)

        # 构建 UI
        self._build_ui()

        # 初始化
        self._refresh_suggestions()
        self._schedule_infer()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = self.ui.pad
        ip = self.ui.inner_pad

        # 顶部：图像 + 工具
        top = ttk.Frame(self, padding=pad)
        top.pack(side=tk.TOP, fill=tk.X)

        self.img_label = ttk.Label(top, text="(生成图在此显示)", style="Title.TLabel")
        self.img_label.pack(side=tk.LEFT, padx=ip)

        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=ip)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self.status_var).pack(anchor="w", pady=(0, ip))

        row1 = ttk.Frame(right)
        row1.pack(anchor="w", pady=self.ui.button_pady)
        ttk.Button(row1, text="Resample z", command=self._resample_z).pack(side=tk.LEFT, padx=self.ui.button_padx)
        ttk.Button(row1, text="Save PNG", command=self._save_png).pack(side=tk.LEFT, padx=self.ui.button_padx)

        row2 = ttk.Frame(right)
        row2.pack(anchor="w", pady=self.ui.button_pady)
        ttk.Button(row2, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=self.ui.button_padx)
        ttk.Button(row2, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=self.ui.button_padx)

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=pad, pady=(4, 6))

        # 中部：前缀显示
        prefix_frame = ttk.Frame(self, padding=(pad, 0))
        prefix_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(prefix_frame, text="Prefix:", style="Title.TLabel").pack(side=tk.LEFT)
        self.prefix_var = tk.StringVar(value="")
        self.prefix_label = ttk.Label(prefix_frame, textvariable=self.prefix_var)
        self.prefix_label.pack(side=tk.LEFT, padx=ip)

        # 底部：候选词 Listbox + 控件
        bottom = ttk.Frame(self, padding=pad)
        bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(bottom)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Next tokens:").pack(anchor="w")
        self.listbox = tk.Listbox(
            left,
            height=self.ui.listbox_height,
            width=self.ui.listbox_width,
            exportselection=False,
            font=self.font_base,
        )
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=yscroll.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.bind("<Double-Button-1>", lambda e: self._add_selected())

        controls = ttk.Frame(bottom)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=ip)

        ttk.Button(controls, text="Add ➜", command=self._add_selected).pack(pady=4, fill=tk.X)

        ttk.Label(controls, text="Manual token:").pack(anchor="w", pady=(8, 2))
        self.manual_entry = ttk.Entry(controls, font=self.font_base)
        self.manual_entry.pack(fill=tk.X)
        self.manual_entry.bind("<Return>", lambda e: self._add_manual())

        ttk.Button(controls, text="Add manual", command=self._add_manual).pack(pady=4, fill=tk.X)

        # 鼠标滚轮：滚动 listbox（Windows/Mac 差异这里保持默认行为）
        # 其他容器滚动需求可以按需绑定 Canvas

    # ---------------- 逻辑 ----------------
    def _refresh_prefix_view(self):
        self.prefix_var.set(" ".join(self.prefix_tokens))

    def _refresh_suggestions(self):
        try:
            candidates = self.trie.next_tokens(self.prefix_tokens)
        except Exception:
            candidates = []
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
        tok = "".join([ch for ch in tok if (ch.isalpha() or ch == " ")])
        tok = tok.strip()
        if not tok:
            return
        self._push_token(tok)
        self.manual_entry.delete(0, tk.END)

    def _push_token(self, tok: str):
        valid = tok in (self.trie.next_tokens(self.prefix_tokens) or [])
        if not valid:
            print(f"[Warn] '{tok}' 不在当前前缀候选中。")
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

    def _update_condition_from_prefix(self):
        self.c.zero_()
        if not self.vocab:
            return
        for tok in self.prefix_tokens:
            idx = self.vocab.get(tok, None)
            if idx is not None and 0 <= idx < self.cond_dim:
                self.c[0, idx] = 1.0

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
            pil_small = pil.resize((self.ui.image_width, self.ui.image_height),
                                   resample=self.ui.pil_resample)
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
            pil_small = pil.resize((self.ui.image_width, self.ui.image_height),
                                   resample=self.ui.pil_resample)
            pil_small.save(file)
            messagebox.showinfo("Saved", f"Saved to {file}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# =========================
# 入口
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    # 模型/数据
    p.add_argument("--ckpt", type=str, default="./models_conv/last.pt")
    p.add_argument("--trie", type=str, default="./caption_trie.json")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--vocab", type=str, default="./vocab.json")
    p.add_argument("--cond_dim", type=int, default=270)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # UI 配置文件
    p.add_argument("--ui_config", type=str, default="./ui.json", help="JSON 文件路径")

    # UI 覆盖项（可不填）
    p.add_argument("--font_family", type=str, default=None)
    p.add_argument("--font_size_base", type=int, default=None)
    p.add_argument("--font_size_title", type=int, default=None)
    p.add_argument("--bold_title", type=int, default=None)  # 1/0

    p.add_argument("--pad", type=int, default=None)
    p.add_argument("--inner_pad", type=int, default=None)
    p.add_argument("--button_padx", type=int, default=None)
    p.add_argument("--button_pady", type=int, default=None)

    p.add_argument("--listbox_height", type=int, default=None)
    p.add_argument("--listbox_width", type=int, default=None)

    p.add_argument("--image_width", type=int, default=None)
    p.add_argument("--image_height", type=int, default=None)
    p.add_argument("--resample", type=str, default=None, help="NEAREST/BILINEAR/BICUBIC/LANCZOS")

    p.add_argument("--theme", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ui = UIConfig.from_json(args.ui_config)
    ui.override_with_args(args)

    app = TrieSelectApp(args, ui)
    app.mainloop()


# python app2.py \
#   --ckpt ./models_conv/last.pt \
#   --trie ./caption_trie.json \
#   --vocab ./vocab.json \
#   --ui_config ./ui.json
