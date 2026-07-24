"""Visualizes the DDPM reverse-process trajectory (x_T noise -> x_0 image) as
a strip of intermediate snapshots — makes the "iterative denoising" story
concrete for the report, rather than just showing the final sample."""
import matplotlib.pyplot as plt
import torch

from src.sampling.ddpm_sampler import ddpm_sample
from src.utils.image_utils import denorm


@torch.no_grad()
def save_denoising_strip(unet, diffusion, img_size, device, save_path):
    _, trajectory = ddpm_sample(unet, diffusion, img_size, batch_size=1, device=device,
                                 return_trajectory=True)
    frames = [denorm(t[0]).permute(1, 2, 0).numpy() for t in trajectory]

    fig, axes = plt.subplots(1, len(frames), figsize=(2 * len(frames), 2))
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    return frames
