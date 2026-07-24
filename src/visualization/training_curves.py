"""Plots and saves loss curves for VAE / DDPM training history."""
import matplotlib.pyplot as plt


def save_curve_plot(history_keys_to_values: dict, title: str, save_path: str, xlabel="epoch"):
    plt.figure(figsize=(7, 5))
    for label, values in history_keys_to_values.items():
        plt.plot(values, label=label)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def save_vae_curves(history: list, save_path: str):
    save_curve_plot(
        {"recon": [h["recon"] for h in history], "kld": [h["kld"] for h in history]},
        "VAE training curves", save_path,
    )


def save_ddpm_curves(history: list, save_path: str):
    save_curve_plot({"loss": [h["loss"] for h in history]}, "DDPM training loss", save_path)
