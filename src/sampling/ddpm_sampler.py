"""Full ancestral DDPM reverse sampling (T sequential steps). Slow but exactly
matches the training-time process — use this for final reported results."""
import torch


@torch.no_grad()
def ddpm_p_sample_step(model, diffusion, x_t, t):
    betas_t = diffusion.extract(diffusion.betas, t, x_t.shape)
    sqrt_omac_t = diffusion.extract(diffusion.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    sqrt_recip_alphas_t = diffusion.extract(1.0 / torch.sqrt(diffusion.alphas), t, x_t.shape)

    pred_noise = model(x_t, t)
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_omac_t)

    if (t == 0).all():
        return model_mean
    posterior_var_t = diffusion.extract(diffusion.posterior_variance, t, x_t.shape)
    noise = torch.randn_like(x_t)
    mask = (t > 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))
    return model_mean + mask * torch.sqrt(posterior_var_t) * noise


@torch.no_grad()
def ddpm_sample(model, diffusion, img_size, batch_size, channels=3, device="cpu", return_trajectory=False):
    """Full reverse process from pure noise x_T to x_0.
    If return_trajectory=True, also returns a list of intermediate x_t
    snapshots (used by src/visualization/denoising.py)."""
    x = torch.randn(batch_size, channels, img_size, img_size, device=device)
    trajectory = [x.cpu()] if return_trajectory else None

    for i in reversed(range(diffusion.T)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        x = ddpm_p_sample_step(model, diffusion, x, t)
        if return_trajectory and i % max(1, diffusion.T // 10) == 0:
            trajectory.append(x.cpu())

    if return_trajectory:
        return x, trajectory
    return x
