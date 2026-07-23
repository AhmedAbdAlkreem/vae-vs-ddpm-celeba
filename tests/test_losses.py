"""Sanity tests for loss functions."""
import torch

from src.losses.diffusion_loss import simple_diffusion_loss
from src.losses.vae_loss import vae_loss


def test_simple_diffusion_loss():
    pred = torch.randn(4, 3, 16, 16)
    true = torch.randn(4, 3, 16, 16)
    loss = simple_diffusion_loss(pred, true)
    assert torch.isfinite(loss)


def test_vae_loss_components():
    recon = torch.randn(4, 3, 16, 16)
    x = torch.randn(4, 3, 16, 16)
    mu = torch.randn(4, 8)
    logvar = torch.randn(4, 8)
    total, recon_l, kld, perceptual = vae_loss(recon, x, mu, logvar, kld_weight=0.5)
    assert torch.isfinite(total)
    assert torch.isfinite(recon_l)
    assert torch.isfinite(kld)
