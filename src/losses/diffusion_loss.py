"""DDPM training loss: simple noise-prediction MSE (L_simple from Ho et al. 2020)."""
import torch.nn.functional as F


def simple_diffusion_loss(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)
