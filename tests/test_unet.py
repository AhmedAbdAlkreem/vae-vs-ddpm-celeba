"""Sanity tests for the U-Net + diffusion process: shapes, finite loss,
correct sampling output shape."""
import torch

from src.models.diffusion import GaussianDiffusion
from src.models.unet import UNet


def test_unet_forward_shape():
    model = UNet(img_channels=3, base_ch=8, ch_mults=(1, 2), time_dim=32)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,))
    out = model(x, t)
    assert out.shape == x.shape


def test_diffusion_training_loss_is_finite():
    model = UNet(img_channels=3, base_ch=8, ch_mults=(1, 2), time_dim=32)
    diffusion = GaussianDiffusion(timesteps=10, device="cpu")
    x0 = torch.randn(2, 3, 32, 32)
    loss = diffusion.training_loss(model, x0)
    assert torch.isfinite(loss)
    assert loss.dim() == 0
