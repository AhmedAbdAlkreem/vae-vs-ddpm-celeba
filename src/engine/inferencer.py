"""
Thin orchestration layer for single-image / single-sample inference: loads
checkpoints and exposes simple encode/decode/generate calls. Used by
inference.py.
"""
import os

import torch
from PIL import Image

from src.datasets.transforms import build_transform
from src.models.diffusion import GaussianDiffusion
from src.models.unet import UNet
from src.models.vae import VAE
from src.sampling.ddim_sampler import ddim_sample
from src.sampling.ddpm_sampler import ddpm_sample
from src.training.ema import EMA
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device
from src.utils.image_utils import denorm


class GenAIInferencer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device()
        self.transform = build_transform(cfg.dataset.img_size, augment=False)

        self.vae = VAE.from_config(cfg).to(self.device)
        vae_ckpt = load_checkpoint(os.path.join(cfg.checkpoint_dir, "vae_best.pt"),
                                    map_location=self.device)
        self.vae.load_state_dict(vae_ckpt["model"])
        self.vae.eval()

        self.unet = UNet.from_config(cfg).to(self.device)
        ddpm_ckpt = load_checkpoint(os.path.join(cfg.checkpoint_dir, "ddpm_best.pt"),
                                     map_location=self.device)
        self.ema = EMA(self.unet, decay=cfg.diffusion.ema_decay)
        self.ema.load_state_dict(ddpm_ckpt["ema"])
        self.ema.copy_to(self.unet)
        self.unet.eval()
        self.diffusion = GaussianDiffusion.from_config(cfg, self.device)

    @torch.no_grad()
    def reconstruct_image(self, image_path: str):
        """VAE encode -> decode a single real image, returns (input, reconstruction) as [0,1] tensors."""
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        recon, _, _ = self.vae(x)
        return denorm(x[0]).cpu(), denorm(recon[0]).cpu()

    @torch.no_grad()
    def generate_one(self, source: str = "ddpm"):
        """Generates a single new image from either model ("vae" or "ddpm")."""
        if source == "vae":
            img = self.vae.sample(1, self.device)[0]
        elif self.cfg.diffusion.sampler == "ddim":
            img = ddim_sample(self.unet, self.diffusion, self.cfg.dataset.img_size, 1,
                               device=self.device, ddim_steps=self.cfg.diffusion.ddim_steps,
                               eta=self.cfg.diffusion.ddim_eta)[0]
        else:
            img = ddpm_sample(self.unet, self.diffusion, self.cfg.dataset.img_size, 1,
                               device=self.device)[0]
        return denorm(img).cpu()
