"""Diffusion-process configuration."""
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ema_decay: float = 0.999

    # Sampler used at inference time: "ddpm" (all T steps, slow, matches training
    # exactly) or "ddim" (fewer steps, deterministic, ~10-20x faster, small
    # quality trade-off). See src/sampling/.
    sampler: str = "ddpm"
    ddim_steps: int = 50
    ddim_eta: float = 0.0
