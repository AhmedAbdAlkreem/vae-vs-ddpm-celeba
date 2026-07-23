#!/usr/bin/env python3
"""
Run the full quantitative + qualitative evaluation suite: FID, Inception
Score, VAE reconstruction metrics, comparison grids, latent-space plot,
interpolation grid, and a DDPM denoising-process strip + GIF.

    python evaluate.py
"""
import json
import os

from configs.config import Config
from src.engine.evaluator import GenAIEvaluator
from src.evaluation.evaluator import Evaluator
from src.utils.paths import ensure_output_dirs
from src.visualization.denoising import save_denoising_strip
from src.visualization.gif import save_gif
from src.visualization.interpolation import save_interpolation_grid
from src.visualization.latent_space import save_latent_space_plot
from src.visualization.reconstruction import save_reconstruction_grid


def main():
    cfg = Config()
    ensure_output_dirs(cfg)

    evaluator = GenAIEvaluator(cfg)
    results, real_imgs, vae_imgs, ddpm_imgs = evaluator.run()
    print(json.dumps(results, indent=2))

    vae = evaluator._load_vae()
    unet, diffusion, ema = evaluator._load_ddpm()
    eval_unet = type(unet)(
        img_channels=cfg.model.img_channels, base_ch=cfg.model.unet_base_ch,
        ch_mults=cfg.model.unet_ch_mults, time_dim=cfg.model.unet_time_dim,
    ).to(evaluator.device)
    ema.copy_to(eval_unet)

    print("Saving reconstruction grid...")
    save_reconstruction_grid(vae, real_imgs, evaluator.device,
                              f"{cfg.output_dir}/reconstructions/vae_reconstructions.png")

    print("Saving latent-space plot...")
    save_latent_space_plot(vae, real_imgs, evaluator.device,
                            f"{cfg.output_dir}/latent_space/vae_latent_pca.png")

    print("Saving latent interpolation grid...")
    save_interpolation_grid(vae, evaluator.device,
                             f"{cfg.output_dir}/interpolation/vae_interpolation.png")

    print("Saving DDPM denoising strip + GIF...")
    frames = save_denoising_strip(eval_unet, diffusion, cfg.dataset.img_size, evaluator.device,
                                   f"{cfg.output_dir}/denoising/ddpm_denoising_strip.png")
    save_gif(frames, f"{cfg.output_dir}/gifs/ddpm_denoising.gif")

    print("All evaluation artifacts saved to", cfg.output_dir)


if __name__ == "__main__":
    main()
