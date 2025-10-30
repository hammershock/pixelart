# inference_conv.py
# inference_conv.py — 卷积CVAE推理：随机(或自定义)条件 -> 采样生成像素图
import os
import math
import argparse
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torchvision.utils import save_image, make_grid
    _HAS_TV = True
except Exception:
    _HAS_TV = False
    from PIL import Image
    import numpy as np

# ============== 与 train_conv.py 对齐的模块 ==============
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
    def __init__(self, img_channels: int, img_size: Tuple[int, int], cond_dim: int,
                 latent_dim: int = 64, base_ch: int = 32):
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

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = x
        for blk in self.enc_blocks:
            h = blk(h)
        h = h.view(h.size(0), -1)
        hc = torch.cat([h, c], dim=-1)
        hc = self.fc_enc(hc)
        mu = self.mu_head(hc)
        logvar = self.logvar_head(hc)
        return mu, logvar

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        zc = torch.cat([z, c], dim=-1)
        h = self.fc_dec(zc)
        h = h.view(h.size(0), self.enc_out_ch, self.enc_out_h, self.enc_out_w)
        for blk in self.dec_blocks:
            h = blk(h)
        x_hat = self.dec_head(h)
        return x_hat

# ============== 条件构造/保存工具 ==============
def make_conditions(mode: str, batch_size: int, cond_dim: int, device, p: float = 0.15, k: int = 8,
                    custom_path: str = None, text: str = None, vocab_path: str = None):
    """
    构造条件向量:
      - bernoulli: 每维以概率 p 置 1
      - k_hot: 每个样本随机 K 个位置置 1
      - zeros/ones: 全0/全1
      - custom_npy: 从 .npy 读取 [B, cond_dim]
      - text: 需要 vocab.json + 文本，tokenize 后复制成 B 份
    """
    if mode == "bernoulli":
        c = (torch.rand(batch_size, cond_dim, device=device) < p).float()
    elif mode == "k_hot":
        c = torch.zeros(batch_size, cond_dim, device=device)
        idx = torch.randint(0, cond_dim, (batch_size, k), device=device)
        row = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(idx)
        c[row, idx] = 1.0
    elif mode == "zeros":
        c = torch.zeros(batch_size, cond_dim, device=device)
    elif mode == "ones":
        c = torch.ones(batch_size, cond_dim, device=device)
    elif mode == "custom_npy":
        assert custom_path is not None, "--custom_path is required for custom_npy"
        import numpy as np
        arr = np.load(custom_path)
        assert arr.ndim == 2 and arr.shape[1] == cond_dim, "custom npy shape must be [B, cond_dim]"
        assert arr.shape[0] == batch_size, "custom npy batch size mismatch"
        c = torch.from_numpy(arr).float().to(device)
    elif mode == "text":
        assert (text is not None) and (vocab_path is not None), "--text and --vocab are required for text mode"
        import json, re, numpy as np
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        t = re.sub(r"[^a-z\s]", "", text.lower()).split()
        vec = np.zeros((cond_dim,), dtype="float32")
        for tok in t:
            if tok in vocab and vocab[tok] < cond_dim:
                vec[vocab[tok]] = 1.0
        c = torch.tensor(vec, device=device).unsqueeze(0).repeat(batch_size, 1)
    else:
        raise ValueError(f"Unknown cond_mode: {mode}")
    return c

@torch.no_grad()
def save_grid(tensor_bchw: torch.Tensor, out_path: str):
    imgs = tensor_bchw.clamp(0, 1).cpu()
    if _HAS_TV:
        n = imgs.size(0)
        nrow = max(1, int(math.sqrt(n)))
        grid = make_grid(imgs, nrow=nrow)
        save_image(grid, out_path)
    else:
        from PIL import Image
        arr = imgs.numpy()
        B, C, H, W = arr.shape
        nrow = max(1, int(math.sqrt(B)))
        ncol = math.ceil(B / nrow)
        canvas = Image.new("RGB", (ncol * W, nrow * H))
        for i in range(B):
            tile = (arr[i].transpose(1, 2, 0) * 255).astype("uint8")
            tile = Image.fromarray(tile)
            r, c = divmod(i, ncol)
            canvas.paste(tile, (c * W, r * H))
        canvas.save(out_path)

# ============== 从ckpt推断超参并采样 ==============
def infer_hparams_from_state_dict(state_dict, img_channels: int):
    # base_ch: 第一个encoder conv的输出通道
    first_conv_w = state_dict["enc_blocks.0.0.weight"]  # [out_ch, in_ch, 3, 3]
    base_ch = int(first_conv_w.shape[0])
    # latent_dim: mu_head.bias 维度
    latent_dim = int(state_dict["mu_head.bias"].shape[0])
    return base_ch, latent_dim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./models_conv/last.pt")
    parser.add_argument("--out", type=str, default="./samples_conv/infer.png")
    parser.add_argument("--batch_size", type=int, default=16, help="一次生成多少张")
    parser.add_argument("--seed", type=int, default=42)
    # 条件构造
    parser.add_argument("--cond_mode", type=str, default="bernoulli",
                        choices=["bernoulli", "k_hot", "zeros", "ones", "custom_npy", "text"])
    parser.add_argument("--cond_dim", type=int, default=270, help="条件维度（默认270）")
    parser.add_argument("--bernoulli_p", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=8, help="k_hot 模式每样本置1的个数")
    parser.add_argument("--custom_path", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--vocab", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载ckpt
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model_state_dict"]
    img_shape = ckpt.get("img_shape", None)  # (C,H,W) — 在 train_conv.py 里已保存
    cond_dim_meta = ckpt.get("cond_dim", None)

    if img_shape is None:
        raise RuntimeError("Checkpoint missing 'img_shape'. Please retrain with train_conv.py that saves it.")
    C, H, W = img_shape

    # 推断 base_ch 与 latent_dim
    base_ch, latent_dim = infer_hparams_from_state_dict(state_dict, img_channels=C)

    # 条件维度：优先用命令行传入，否则回退ckpt里的
    cond_dim = args.cond_dim if args.cond_dim is not None else cond_dim_meta
    if cond_dim_meta is not None and cond_dim != cond_dim_meta:
        print(f"[Warn] cond_dim mismatch: ckpt={cond_dim_meta}, cli={cond_dim}. "
              f"Using cli value {cond_dim}. Ensure it matches training vocab size!")

    # 构建模型并加载权重
    model = CVAEConv(
        img_channels=C,
        img_size=(H, W),
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        base_ch=base_ch,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 构造条件向量（默认：随机Bernoulli，270维）
    c = make_conditions(
        mode=args.cond_mode,
        batch_size=args.batch_size,
        cond_dim=cond_dim,
        device=device,
        p=args.bernoulli_p,
        k=args.k,
        custom_path=args.custom_path,
        text=args.text,
        vocab_path=args.vocab,
    )

    # 采样 z ~ N(0, I) 并解码
    with torch.no_grad():
        z = torch.randn(args.batch_size, latent_dim, device=device)
        imgs = model.decode(z, c)  # [B, C, H, W] in [0,1]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_grid(imgs, args.out)
    print(f"Saved {args.batch_size} samples to: {args.out}")
    print(f"(img_shape={img_shape}, latent_dim={latent_dim}, base_ch={base_ch}, cond_dim={cond_dim})")


if __name__ == "__main__":
    main()
