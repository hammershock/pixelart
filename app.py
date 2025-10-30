# app.py
# build with tk
# =================
# 上侧IMAGE DISPLAY，resize到128*128（像素风使用NEAREST）
# 下侧为270维条件的勾选框网格（可滚动），切换即运行推理并更新图像
# ================

import os
import math
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageTk

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
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))  # 保像素味
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

# 从 state_dict 推断 base_ch / latent_dim
def infer_hparams_from_state_dict(state_dict):
    base_ch = int(state_dict["enc_blocks.0.0.weight"].shape[0])
    latent_dim = int(state_dict["mu_head.bias"].shape[0])
    return base_ch, latent_dim

# --------------------------
# Tk App
# --------------------------
class PixelCVAEApp(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.title("Pixel CVAE - Interactive Inference")
        self.args = args

        # 设备
        self.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        torch.manual_seed(args.seed)

        # 加载 ckpt
        ckpt = torch.load(args.ckpt, map_location=self.device)
        self.state_dict = ckpt["model_state_dict"]
        if "img_shape" not in ckpt:
            messagebox.showerror("Error", "Checkpoint missing 'img_shape'. 请用 train_conv.py 重新训练保存。")
            raise SystemExit(1)
        self.C, self.H, self.W = ckpt["img_shape"]
        self.cond_dim_meta = ckpt.get("cond_dim", None)

        # 条件维度（默认270，可通过命令行覆盖）
        self.cond_dim = args.cond_dim if args.cond_dim is not None else (self.cond_dim_meta or 270)
        if self.cond_dim_meta is not None and self.cond_dim != self.cond_dim_meta:
            print(f"[Warn] cond_dim mismatch: ckpt={self.cond_dim_meta}, cli={self.cond_dim}. "
                  f"使用 cli 值 {self.cond_dim}（请确保与训练一致）。")

        # 推断结构
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

        # 准备一个固定的 z（可按按钮重采样）
        self.z = torch.randn(1, self.latent_dim, device=self.device)

        # 条件初始化（Bernoulli）
        self.c = (torch.rand(1, self.cond_dim, device=self.device) < args.init_p).float()

        # 标签名（可选）
        self.labels = [f"c{i}" for i in range(self.cond_dim)]
        if args.vocab and os.path.exists(args.vocab):
            try:
                import json
                with open(args.vocab, "r") as f:
                    vocab = json.load(f)  # token -> idx
                inv = [None] * self.cond_dim
                for tok, idx in vocab.items():
                    if 0 <= idx < self.cond_dim:
                        inv[idx] = tok
                for i in range(self.cond_dim):
                    if inv[i] is not None:
                        self.labels[i] = inv[i]
            except Exception as e:
                print(f"[Warn] 读取 vocab 失败：{e}")

        # UI
        self._build_ui()

        # 初始推理并显示
        self._schedule_infer()

    def _build_ui(self):
        # 顶部图像显示区
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.img_label = ttk.Label(top, text="(生成图将显示在此)")
        self.img_label.pack(side=tk.LEFT, padx=8)

        right_tools = ttk.Frame(top)
        right_tools.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(right_tools, textvariable=self.status_var).pack(anchor="w", pady=(0, 6))

        btn_row = ttk.Frame(right_tools)
        btn_row.pack(anchor="w", pady=2)
        ttk.Button(btn_row, text="Resample z", command=self._resample_z).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Save PNG", command=self._save_png).pack(side=tk.LEFT, padx=2)

        btn_row2 = ttk.Frame(right_tools)
        btn_row2.pack(anchor="w", pady=2)
        ttk.Button(btn_row2, text="All On", command=lambda: self._set_all(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="All Off", command=lambda: self._set_all(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="Random", command=self._randomize_conditions).pack(side=tk.LEFT, padx=2)

        # 分隔线
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=8, pady=(4, 6))

        # 可滚动条件面板
        container = ttk.Frame(self)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        self.canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.cond_frame = ttk.Frame(self.canvas)

        self.cond_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.cond_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=vsb.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # 构建复选框网格
        self.vars = []
        N_COLS = self.args.cols if self.args.cols is not None else 10
        for i in range(self.cond_dim):
            var = tk.IntVar(value=int(self.c[0, i].item() > 0.5))
            self.vars.append(var)
            cb = ttk.Checkbutton(self.cond_frame, text=self.labels[i], variable=var,
                                 command=self._on_cond_changed)
            r, c = divmod(i, N_COLS)
            cb.grid(row=r, column=c, sticky="w", padx=6, pady=4)

        # 让网格更紧凑
        for c in range(N_COLS):
            self.cond_frame.grid_columnconfigure(c, weight=1)

    # 事件：鼠标滚轮滚动
    def _on_mousewheel(self, event):
        # Windows 正负方向不同，这里统一
        delta = -1 * (event.delta // 120)
        self.canvas.yview_scroll(delta, "units")

    # 条件变化回调：做一次去抖，避免频繁刷新
    def _on_cond_changed(self):
        # 从复选框同步到 self.c
        for i, var in enumerate(self.vars):
            self.c[0, i] = float(var.get())
        self._schedule_infer()

    def _schedule_infer(self, delay_ms=50):
        # 简单防抖
        if hasattr(self, "_infer_job") and self._infer_job is not None:
            self.after_cancel(self._infer_job)
        self._infer_job = self.after(delay_ms, self._infer_and_update)

    def _infer_and_update(self):
        self._infer_job = None
        try:
            self.status_var.set("Generating...")
            self.update_idletasks()
            with torch.no_grad():
                imgs = self.model.decode(self.z, self.c.to(self.device))  # [1, C, H, W]
            img = imgs[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            pil = Image.fromarray((img * 255).astype(np.uint8))
            # resize到128x128，像素风用NEAREST
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

    def _set_all(self, v: int):
        for i, var in enumerate(self.vars):
            var.set(v)
            self.c[0, i] = float(v)
        self._schedule_infer()

    def _randomize_conditions(self):
        p = self.args.init_p
        rnd = (torch.rand(self.cond_dim) < p).int().tolist()
        for i, var in enumerate(self.vars):
            var.set(rnd[i])
            self.c[0, i] = float(rnd[i])
        self._schedule_infer()

    def _save_png(self):
        # 保存当前 128x128 显示图
        try:
            file = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")],
                initialfile="cvae_sample.png",
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
    p.add_argument("--vocab", type=str, default="vocab.json", help="可选：vocab.json，用于展示条件名称")
    p.add_argument("--cond_dim", type=int, default=270, help="条件维度（默认270）")
    p.add_argument("--init_p", type=float, default=0.15, help="随机初始化条件为1的概率")
    p.add_argument("--cols", type=int, default=10, help="条件网格的列数")
    p.add_argument("--cpu", action="store_true", help="强制使用CPU推理")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    app = PixelCVAEApp(args)
    app.mainloop()
