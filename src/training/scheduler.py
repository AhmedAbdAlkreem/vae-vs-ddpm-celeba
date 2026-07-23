"""Optional learning-rate schedulers. Disabled (constant LR) by default —
the VAE/DDPM configs above train fine with a fixed LR, but a cosine decay is
provided since Kaggle GPU-time-limited runs sometimes benefit from LR decay
in the last few epochs."""
import torch


def build_cosine_scheduler(optimizer, total_epochs: int, enabled: bool = False):
    if not enabled:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
