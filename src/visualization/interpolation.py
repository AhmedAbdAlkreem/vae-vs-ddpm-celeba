"""Latent-space interpolation grid: decode a linear path between two random
latent vectors — a classic qualitative check that the VAE latent space is
smooth/meaningful rather than degenerate."""
import matplotlib.pyplot as plt
import torch

from src.sampling.latent_sampler import interpolate_latents
from src.utils.image_utils import denorm


@torch.no_grad()
def save_interpolation_grid(vae, device, save_path, steps=10, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    z_start = torch.randn(1, vae.latent_dim, device=device, generator=g)[0]
    z_end = torch.randn(1, vae.latent_dim, device=device, generator=g)[0]

    imgs = interpolate_latents(vae, z_start, z_end, steps=steps)

    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2))
    for i in range(steps):
        axes[i].imshow(denorm(imgs[i]).cpu().permute(1, 2, 0).numpy())
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
