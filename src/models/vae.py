"""Variational Autoencoder: wires together Encoder + Decoder + reparameterization."""
import torch
import torch.nn as nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class VAE(nn.Module):
    def __init__(self, img_channels=3, base_ch=64, latent_dim=128, img_size=64):
        super().__init__()
        self.encoder = Encoder(img_channels, base_ch, latent_dim, img_size)
        self.decoder = Decoder(img_channels, base_ch, latent_dim, img_size)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode(self, x):
        """Deterministic encode (returns mu), used for reconstruction /
        latent-space visualizations rather than stochastic sampling."""
        mu, logvar = self.encoder(x)
        return mu, logvar

    @torch.no_grad()
    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    @classmethod
    def from_config(cls, cfg):
        return cls(
            img_channels=cfg.model.img_channels,
            base_ch=cfg.model.vae_base_ch,
            latent_dim=cfg.model.latent_dim,
            img_size=cfg.dataset.img_size,
        )
