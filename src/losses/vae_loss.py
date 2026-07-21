"""VAE objective: reconstruction + KL divergence (+ optional perceptual term)."""
import torch
import torch.nn.functional as F


def vae_loss(recon, x, mu, logvar, kld_weight=1.0, perceptual_loss_fn=None, perceptual_weight=0.0):
    """Returns (total_loss, recon_loss, kld, perceptual_loss_value).
    All terms are averaged per-sample. Pixels are assumed to be in [-1, 1]."""
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon_loss + kld_weight * kld

    perceptual_value = torch.tensor(0.0, device=x.device)
    if perceptual_loss_fn is not None and perceptual_weight > 0:
        perceptual_value = perceptual_loss_fn(recon, x)
        total = total + perceptual_weight * perceptual_value

    return total, recon_loss, kld, perceptual_value
