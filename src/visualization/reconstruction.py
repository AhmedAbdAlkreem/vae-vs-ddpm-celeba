"""Side-by-side real vs. VAE-reconstructed image grid — visual complement to
src/evaluation/reconstruction.py's numeric MSE/PSNR."""
import matplotlib.pyplot as plt
import torch

from src.utils.image_utils import denorm


@torch.no_grad()
def save_reconstruction_grid(vae, real_batch, device, save_path, n=8):
    vae.eval()
    x = real_batch[:n].to(device)
    recon, _, _ = vae(x)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(denorm(x[i]).cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(denorm(recon[i]).cpu().permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Real", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
