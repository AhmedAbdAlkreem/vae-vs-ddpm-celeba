"""VAE-specific reconstruction-quality metrics (only meaningful for the VAE,
since DDPM has no encoder / explicit reconstruction step)."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_reconstruction_metrics(vae, real_batch, device):
    vae.eval()
    x = real_batch.to(device)
    recon, mu, logvar = vae(x)

    mse = F.mse_loss(recon, x).item()
    # PSNR assuming pixel range [-1, 1] -> data range 2.0
    psnr = 10 * torch.log10(torch.tensor(2.0 ** 2) / max(mse, 1e-10)).item()

    return {"reconstruction_mse": mse, "psnr_db": psnr}
