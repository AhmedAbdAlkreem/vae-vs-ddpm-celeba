"""Sanity tests for DDPM and DDIM samplers."""
import torch

from src.models.diffusion import GaussianDiffusion
from src.models.unet import UNet
from src.sampling.ddim_sampler import ddim_sample
from src.sampling.ddpm_sampler import ddpm_sample


def _tiny_setup():
    model = UNet(img_channels=3, base_ch=8, ch_mults=(1, 2), time_dim=32)
    diffusion = GaussianDiffusion(timesteps=10, device="cpu")
    return model, diffusion


def test_ddpm_sample_shape():
    model, diffusion = _tiny_setup()
    out = ddpm_sample(model, diffusion, img_size=32, batch_size=2, device="cpu")
    assert out.shape == (2, 3, 32, 32)


def test_ddim_sample_shape_and_fewer_steps():
    model, diffusion = _tiny_setup()
    out = ddim_sample(model, diffusion, img_size=32, batch_size=2, device="cpu", ddim_steps=5)
    assert out.shape == (2, 3, 32, 32)
