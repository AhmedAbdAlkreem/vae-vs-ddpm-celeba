"""2D projection (PCA) of VAE latent codes for a batch of real images — a
sanity check that the learned latent space has meaningful structure rather
than collapsing to a single point (posterior collapse)."""
import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def save_latent_space_plot(vae, real_batch, device, save_path):
    from sklearn.decomposition import PCA

    vae.eval()
    x = real_batch.to(device)
    mu, _ = vae.encode(x)
    mu = mu.cpu().numpy()

    if mu.shape[1] > 2:
        mu_2d = PCA(n_components=2).fit_transform(mu)
    else:
        mu_2d = mu

    plt.figure(figsize=(6, 6))
    plt.scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.6, s=15)
    plt.title("VAE latent space (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
