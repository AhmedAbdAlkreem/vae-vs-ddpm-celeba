"""VAE convolutional encoder."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """img_size x img_size x img_channels -> latent_dim (mu, logvar)."""

    def __init__(self, img_channels=3, base_ch=64, latent_dim=128, img_size=64):
        super().__init__()
        n_down = int(torch.log2(torch.tensor(img_size // 4)).item())  # downsample until 4x4
        chs = [img_channels] + [base_ch * (2 ** i) for i in range(n_down)]
        layers = [ConvBlock(chs[i], chs[i + 1]) for i in range(n_down)]
        self.conv = nn.Sequential(*layers)
        self.final_ch = chs[-1]
        self.spatial = 4
        flat_dim = self.final_ch * self.spatial * self.spatial
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu = self.fc_mu(h)
        # Clamp logvar for numerical stability (prevents the KL blow-up that
        # can happen in the first epoch before KL-annealing kicks in).
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 10.0)
        return mu, logvar
