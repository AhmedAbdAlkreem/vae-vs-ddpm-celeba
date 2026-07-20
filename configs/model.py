"""Model architecture configuration (VAE + U-Net)."""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    img_channels: int = 3

    # VAE
    latent_dim: int = 128
    vae_base_ch: int = 64

    # U-Net (noise-prediction network for DDPM)
    unet_base_ch: int = 64
    unet_ch_mults: Tuple[int, ...] = (1, 2, 4, 4)
    unet_time_dim: int = 256
    unet_attn_heads: int = 4
