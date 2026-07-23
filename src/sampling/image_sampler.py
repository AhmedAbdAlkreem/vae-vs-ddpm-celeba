"""
High-level sampling API used by sample.py / inference.py: given trained
model artifacts, generate N new images from either model without the
caller needing to know about GaussianDiffusion / EMA / sampler choice.
"""
import torch

from src.sampling.ddim_sampler import ddim_sample
from src.sampling.ddpm_sampler import ddpm_sample
from src.sampling.latent_sampler import sample_latents


@torch.no_grad()
def generate_vae_samples(vae, n, device, batch=128):
    vae.eval()
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        out.append(sample_latents(vae, b, device).cpu())
    return torch.cat(out)


@torch.no_grad()
def generate_ddpm_samples(unet, diffusion, cfg, n, device, batch=64):
    unet.eval()
    out = []
    sampler = cfg.diffusion.sampler
    for i in range(0, n, batch):
        b = min(batch, n - i)
        if sampler == "ddim":
            imgs = ddim_sample(unet, diffusion, cfg.dataset.img_size, b, device=device,
                                ddim_steps=cfg.diffusion.ddim_steps, eta=cfg.diffusion.ddim_eta)
        else:
            imgs = ddpm_sample(unet, diffusion, cfg.dataset.img_size, b, device=device)
        out.append(imgs.cpu())
    return torch.cat(out)
