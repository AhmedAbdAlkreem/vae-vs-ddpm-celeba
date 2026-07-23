"""
Thin orchestration layer for evaluation: loads checkpoints if models aren't
already in memory, then delegates all metric computation to
src/evaluation/evaluator.py. This is the entry point evaluate.py calls.
"""
import os

from src.datasets.dataloader import get_dataloader
from src.evaluation.evaluator import Evaluator
from src.models.diffusion import GaussianDiffusion
from src.models.unet import UNet
from src.models.vae import VAE
from src.training.ema import EMA
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device


class GenAIEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device()

    def _load_vae(self):
        vae = VAE.from_config(self.cfg).to(self.device)
        ckpt = load_checkpoint(os.path.join(self.cfg.checkpoint_dir, "vae_best.pt"),
                                map_location=self.device)
        vae.load_state_dict(ckpt["model"])
        return vae

    def _load_ddpm(self):
        unet = UNet.from_config(self.cfg).to(self.device)
        ckpt = load_checkpoint(os.path.join(self.cfg.checkpoint_dir, "ddpm_best.pt"),
                                map_location=self.device)
        ema = EMA(unet, decay=self.cfg.diffusion.ema_decay)
        ema.load_state_dict(ckpt["ema"])
        diffusion = GaussianDiffusion.from_config(self.cfg, self.device)
        return unet, diffusion, ema

    def run(self, vae=None, unet=None, diffusion=None, ema=None, files=None, transform=None):
        if files is None or transform is None:
            _, files, transform = get_dataloader(self.cfg)
        if vae is None:
            vae = self._load_vae()
        if unet is None or diffusion is None or ema is None:
            unet, diffusion, ema = self._load_ddpm()

        evaluator = Evaluator(self.cfg, self.device)
        return evaluator.evaluate(vae, unet, diffusion, ema, files, transform)
