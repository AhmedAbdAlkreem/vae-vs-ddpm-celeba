"""Image tensor conversion helpers, shared across training/sampling/evaluation."""
import torch
from torchvision.utils import make_grid


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float tensor -> [0, 1] float tensor, clamped."""
    return (x.clamp(-1, 1) + 1) / 2


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float tensor -> [0, 255] uint8 tensor (for FID / Inception Score)."""
    return (denorm(x) * 255).round().to(torch.uint8)


def to_grid(x: torch.Tensor, nrow: int = 4) -> torch.Tensor:
    """[-1, 1] float batch -> a single [0,1] grid image tensor (C,H,W)."""
    return make_grid(denorm(x), nrow=nrow)
