"""
DDIM sampler (Song et al. 2020): a deterministic (or partially stochastic,
via `eta`), non-Markovian reverse process that reuses a DDPM-trained model
but needs far fewer steps (e.g. 50 instead of 1000) — directly addresses the
"DDPM sampling is slow" bottleneck at some sample-quality cost.
"""
import torch


@torch.no_grad()
def ddim_sample(model, diffusion, img_size, batch_size, channels=3, device="cpu",
                 ddim_steps=50, eta=0.0):
    step_indices = torch.linspace(0, diffusion.T - 1, ddim_steps, device=device).long()
    step_indices = torch.flip(step_indices, dims=[0])  # T-1 -> 0

    x = torch.randn(batch_size, channels, img_size, img_size, device=device)

    for i, t_val in enumerate(step_indices):
        t = torch.full((batch_size,), t_val.item(), device=device, dtype=torch.long)
        pred_noise = model(x, t)

        alpha_t = diffusion.extract(diffusion.alphas_cumprod, t, x.shape)
        if i < len(step_indices) - 1:
            t_prev_val = step_indices[i + 1]
            t_prev = torch.full((batch_size,), t_prev_val.item(), device=device, dtype=torch.long)
            alpha_prev = diffusion.extract(diffusion.alphas_cumprod, t_prev, x.shape)
        else:
            alpha_prev = torch.ones_like(alpha_t)

        # Predicted x0 from the current noise estimate
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-1, 1)

        sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        dir_xt = torch.sqrt((1 - alpha_prev - sigma_t ** 2).clamp(min=0)) * pred_noise
        noise = torch.randn_like(x) if eta > 0 else 0.0

        x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma_t * noise

    return x
