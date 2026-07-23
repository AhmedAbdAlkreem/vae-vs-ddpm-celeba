"""
Orchestrates the full quantitative evaluation: generates samples from both
models, computes FID / Inception Score / VAE reconstruction metrics, and
saves the results + comparison grids to disk.
"""
import json
import os

import numpy as np
import torch
from PIL import Image

from src.evaluation.fid import compute_fid
from src.evaluation.quality import compute_inception_score
from src.evaluation.reconstruction import compute_reconstruction_metrics
from src.sampling.image_sampler import generate_ddpm_samples, generate_vae_samples
from src.visualization.sampling_grid import save_comparison_grid, save_grid


class Evaluator:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def _get_real_batch(self, files, transform, n, seed=0):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
        imgs = [transform(Image.open(files[i]).convert("RGB")) for i in idx]
        return torch.stack(imgs)

    def evaluate(self, vae, unet, diffusion, ema, files, transform):
        cfg = self.cfg
        device = self.device

        eval_unet = type(unet)(
            img_channels=cfg.model.img_channels, base_ch=cfg.model.unet_base_ch,
            ch_mults=cfg.model.unet_ch_mults, time_dim=cfg.model.unet_time_dim,
        ).to(device)
        ema.copy_to(eval_unet)
        eval_unet.eval()

        print("Generating real reference batch...")
        real_imgs = self._get_real_batch(files, transform, cfg.n_eval_samples, seed=cfg.seed)
        print("Generating VAE samples...")
        vae_imgs = generate_vae_samples(vae, cfg.n_eval_samples, device, batch=cfg.eval_batch_size)
        print(f"Generating DDPM samples (sampler={cfg.diffusion.sampler})...")
        ddpm_imgs = generate_ddpm_samples(eval_unet, diffusion, cfg, cfg.n_eval_samples, device,
                                           batch=cfg.eval_batch_size)

        print("Computing FID / Inception Score...")
        vae_fid = compute_fid(real_imgs, vae_imgs, device)
        vae_is_mean, vae_is_std = compute_inception_score(vae_imgs, device)
        ddpm_fid = compute_fid(real_imgs, ddpm_imgs, device)
        ddpm_is_mean, ddpm_is_std = compute_inception_score(ddpm_imgs, device)

        recon_metrics = compute_reconstruction_metrics(vae, real_imgs[:64], device)

        results = {
            "vae": {"fid": vae_fid, "is_mean": vae_is_mean, "is_std": vae_is_std, **recon_metrics},
            "ddpm": {"fid": ddpm_fid, "is_mean": ddpm_is_mean, "is_std": ddpm_is_std,
                     "sampler": cfg.diffusion.sampler},
            "n_eval_samples": cfg.n_eval_samples,
        }

        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        save_grid(real_imgs, "Real CelebA images", os.path.join(cfg.output_dir, "real_grid.png"))
        save_grid(vae_imgs, "VAE samples", os.path.join(cfg.output_dir, "vae_grid.png"))
        save_grid(ddpm_imgs, "DDPM samples", os.path.join(cfg.output_dir, "ddpm_grid.png"))
        save_comparison_grid(real_imgs, vae_imgs, ddpm_imgs,
                              os.path.join(cfg.output_dir, "comparison_grid.png"))

        return results, real_imgs, vae_imgs, ddpm_imgs
