"""Real / VAE / DDPM sample-grid visualizations. Uses whichever matplotlib
backend is already active (inline in a notebook, Agg/etc. in a script) —
saving via plt.savefig() works regardless, so we don't force a backend here."""
import matplotlib.pyplot as plt

from src.utils.image_utils import to_grid


def save_grid(imgs, title, save_path, n=16, nrow=4):
    grid = to_grid(imgs[:n], nrow=nrow)
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def save_comparison_grid(real, vae_imgs, ddpm_imgs, save_path, n=9):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, imgs, title in zip(axes, [real, vae_imgs, ddpm_imgs], ["Real", "VAE", "DDPM"]):
        grid = to_grid(imgs[:n], nrow=3)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
