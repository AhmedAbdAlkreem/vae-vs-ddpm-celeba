"""
Thin orchestration layer: runs the VAE trainer then the DDPM trainer using
one shared dataloader. Contains no training logic itself — that lives in
src/training/train_vae.py and src/training/train_ddpm.py. This is the single
entry point train.py calls.
"""
from src.datasets.dataloader import get_dataloader
from src.training.train_ddpm import train_ddpm
from src.training.train_vae import train_vae
from src.utils.paths import ensure_output_dirs


class GenAITrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        ensure_output_dirs(cfg)
        self.loader, self.files, self.eval_transform = get_dataloader(cfg)

    def run(self):
        print("=" * 60); print("Training VAE"); print("=" * 60)
        vae, vae_history = train_vae(self.cfg, loader=self.loader)

        print("=" * 60); print("Training DDPM"); print("=" * 60)
        unet, diffusion, ema, ddpm_history = train_ddpm(self.cfg, loader=self.loader)

        return {
            "vae": vae, "vae_history": vae_history,
            "unet": unet, "diffusion": diffusion, "ema": ema, "ddpm_history": ddpm_history,
            "files": self.files, "eval_transform": self.eval_transform,
        }
