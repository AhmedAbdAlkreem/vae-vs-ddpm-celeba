"""Sanity tests for the VAE: correct output shapes, valid loss, working sample()."""
import torch

from src.losses.vae_loss import vae_loss
from src.models.vae import VAE


def test_vae_forward_shapes():
    x = torch.randn(4, 3, 64, 64)
    model = VAE(img_channels=3, base_ch=16, latent_dim=32, img_size=64)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (4, 32)
    assert logvar.shape == (4, 32)


def test_vae_loss_is_finite_scalar():
    x = torch.randn(4, 3, 64, 64)
    model = VAE(img_channels=3, base_ch=16, latent_dim=32, img_size=64)
    recon, mu, logvar = model(x)
    total, recon_l, kld, perceptual = vae_loss(recon, x, mu, logvar)
    assert torch.isfinite(total)
    assert total.dim() == 0


def test_vae_sample_shape():
    model = VAE(img_channels=3, base_ch=16, latent_dim=32, img_size=64)
    samples = model.sample(5, "cpu")
    assert samples.shape == (5, 3, 64, 64)
