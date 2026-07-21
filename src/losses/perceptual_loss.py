"""
Optional VGG-based perceptual loss for the VAE. Comparing reconstructions in
a pretrained feature space (rather than raw pixels) is a well-known way to
reduce VAE blur. Off by default (configs/training.py: use_perceptual_loss)
since it adds a ~15M-parameter frozen VGG16 to memory/compute.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):
    def __init__(self, layer_idx: int = 16, device="cpu"):
        """layer_idx=16 corresponds to roughly the relu3_3 feature map of VGG16,
        a common choice for perceptual losses (mid-level texture/structure)."""
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[: layer_idx + 1]
        self.vgg = vgg.to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x):
        # x is in [-1, 1] -> [0, 1] -> ImageNet-normalized, as VGG expects.
        x = (x.clamp(-1, 1) + 1) / 2
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, recon, target):
        f_recon = self.vgg(self._preprocess(recon))
        f_target = self.vgg(self._preprocess(target))
        return F.mse_loss(f_recon, f_target)
