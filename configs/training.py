"""Training-loop configuration."""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    vae_epochs: int = 30
    vae_lr: float = 2e-4
    kl_warmup_epochs: int = 5      # linear KL-annealing, reduces posterior collapse

    ddpm_epochs: int = 40
    ddpm_lr: float = 2e-4

    checkpoint_every: int = 2      # epochs between resumable checkpoints
    use_amp: bool = True           # mixed precision (only active on CUDA)

    use_perceptual_loss: bool = False   # adds VGG perceptual loss to the VAE objective
    perceptual_weight: float = 0.1
