#!/usr/bin/env python3
"""
Generate new images from the trained VAE and/or DDPM.

    python sample.py --model ddpm --n 16
    python sample.py --model vae --n 16
    python sample.py --model both --n 16
"""
import argparse
import os

import torch
from torchvision.utils import save_image

from configs.config import Config
from src.engine.evaluator import GenAIEvaluator
from src.sampling.image_sampler import generate_ddpm_samples, generate_vae_samples
from src.utils.image_utils import denorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ddpm", "both"], default="both")
    parser.add_argument("--n", type=int, default=16)
    args = parser.parse_args()

    cfg = Config()
    evaluator = GenAIEvaluator(cfg)
    os.makedirs(cfg.samples_dir, exist_ok=True)

    if args.model in ("vae", "both"):
        vae = evaluator._load_vae()
        imgs = generate_vae_samples(vae, args.n, evaluator.device)
        for i, img in enumerate(imgs):
            save_image(denorm(img), os.path.join(cfg.samples_dir, f"vae_{i:03d}.png"))
        print(f"Saved {args.n} VAE samples to {cfg.samples_dir}")

    if args.model in ("ddpm", "both"):
        unet, diffusion, ema = evaluator._load_ddpm()
        eval_unet = type(unet)(
            img_channels=cfg.model.img_channels, base_ch=cfg.model.unet_base_ch,
            ch_mults=cfg.model.unet_ch_mults, time_dim=cfg.model.unet_time_dim,
        ).to(evaluator.device)
        ema.copy_to(eval_unet)
        imgs = generate_ddpm_samples(eval_unet, diffusion, cfg, args.n, evaluator.device)
        for i, img in enumerate(imgs):
            save_image(denorm(img), os.path.join(cfg.samples_dir, f"ddpm_{i:03d}.png"))
        print(f"Saved {args.n} DDPM samples to {cfg.samples_dir}")


if __name__ == "__main__":
    main()
