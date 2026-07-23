"""VAE latent-space sampling helpers."""
import torch


@torch.no_grad()
def sample_latents(vae, n, device):
    """Draw n latent vectors from the VAE prior N(0, I) and decode them."""
    return vae.sample(n, device)


@torch.no_grad()
def interpolate_latents(vae, z_start, z_end, steps=10):
    """Linear interpolation between two latent vectors, decoded at each step.
    Used by src/visualization/interpolation.py."""
    alphas = torch.linspace(0, 1, steps, device=z_start.device)
    zs = torch.stack([(1 - a) * z_start + a * z_end for a in alphas])
    return vae.decode(zs)
