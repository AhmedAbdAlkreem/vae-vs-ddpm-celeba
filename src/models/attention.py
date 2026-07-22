"""Self-attention block used inside the U-Net at low spatial resolution."""
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.mha = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x).reshape(b, c, h * w).transpose(1, 2)  # (b, hw, c)
        out, _ = self.mha(h_, h_, h_)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + out
