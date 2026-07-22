"""
U-Net noise-prediction network for DDPM: sinusoidal timestep embeddings,
residual blocks conditioned on time, self-attention at the lowest spatial
resolution (src/models/attention.py), and strided-conv down/up-sampling.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import SelfAttention


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    """Predicts the noise epsilon added to x_t. ~4 resolution levels with
    attention at the lowest resolution."""

    def __init__(self, img_channels=3, base_ch=64, ch_mults=(1, 2, 4, 4), time_dim=256, attn_heads=4):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(img_channels, base_ch, 3, padding=1)

        chs = [base_ch * m for m in ch_mults]
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        in_ch = base_ch
        self.skip_chs = [base_ch]
        for i, out_ch in enumerate(chs):
            self.down_blocks.append(ResBlock(in_ch, out_ch, time_dim))
            in_ch = out_ch
            self.skip_chs.append(in_ch)
            self.down_samples.append(Downsample(in_ch) if i < len(chs) - 1 else nn.Identity())

        self.mid_block1 = ResBlock(in_ch, in_ch, time_dim)
        self.mid_attn = SelfAttention(in_ch, num_heads=attn_heads)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        rev_chs = list(reversed(chs))
        for i, out_ch in enumerate(rev_chs):
            skip_ch = self.skip_chs.pop()
            self.up_blocks.append(ResBlock(in_ch + skip_ch, out_ch, time_dim))
            in_ch = out_ch
            self.up_samples.append(Upsample(in_ch) if i < len(rev_chs) - 1 else nn.Identity())

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_conv = nn.Conv2d(in_ch, img_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.in_conv(x)
        skips = [h]
        for block, down in zip(self.down_blocks, self.down_samples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for block, up in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            h = up(h)

        return self.out_conv(self.act(self.out_norm(h)))

    @classmethod
    def from_config(cls, cfg):
        return cls(
            img_channels=cfg.model.img_channels,
            base_ch=cfg.model.unet_base_ch,
            ch_mults=cfg.model.unet_ch_mults,
            time_dim=cfg.model.unet_time_dim,
            attn_heads=cfg.model.unet_attn_heads,
        )
