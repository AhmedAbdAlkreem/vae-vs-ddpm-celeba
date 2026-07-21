"""VAE convolutional decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, activation="relu"):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = activation

    def forward(self, x):
        x = self.deconv(x)
        if self.activation == "relu":
            x = F.relu(self.bn(x), inplace=True)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        return x


class Decoder(nn.Module):
    """latent_dim -> img_size x img_size x img_channels."""

    def __init__(self, img_channels=3, base_ch=64, latent_dim=128, img_size=64):
        super().__init__()
        n_up = int(torch.log2(torch.tensor(img_size // 4)).item())
        chs = [base_ch * (2 ** i) for i in range(n_up)][::-1]  # e.g. [512,256,128,64]
        self.final_ch = chs[0]
        self.spatial = 4
        flat_dim = self.final_ch * self.spatial * self.spatial
        self.fc = nn.Linear(latent_dim, flat_dim)

        chs = chs + [img_channels]
        layers = [DeconvBlock(chs[i], chs[i + 1], activation="relu") for i in range(n_up - 1)]
        layers.append(DeconvBlock(chs[-2], chs[-1], activation="tanh"))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z).view(-1, self.final_ch, self.spatial, self.spatial)
        return self.deconv(h)
