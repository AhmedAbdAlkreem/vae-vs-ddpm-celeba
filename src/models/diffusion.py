"""
Gaussian diffusion process (Ho et al. 2020): precomputed beta/alpha schedules,
the forward (noising) process q_sample, and the training loss. Reverse
sampling lives in src/sampling/ (ddpm_sampler.py, ddim_sampler.py) since
sampling strategy is a separate concern from the noise schedule itself.
"""
import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def extract(self, arr, t, shape):
        out = arr.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """Forward process: sample x_t given x_0 and timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omac * noise, noise

    def training_loss(self, model, x0, loss_fn=None):
        """Noise-prediction loss. `loss_fn` defaults to simple MSE
        (see src/losses/diffusion_loss.py) but can be swapped."""
        from src.losses.diffusion_loss import simple_diffusion_loss
        loss_fn = loss_fn or simple_diffusion_loss

        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device).long()
        x_t, noise = self.q_sample(x0, t)
        pred_noise = model(x_t, t)
        return loss_fn(pred_noise, noise)

    @classmethod
    def from_config(cls, cfg, device):
        return cls(
            timesteps=cfg.diffusion.timesteps,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            device=device,
        )
