"""Filesystem path helpers."""
import os


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_output_dirs(cfg):
    """Creates the standard outputs/ subfolder layout for a given Config."""
    for sub in ["checkpoints", "logs", "samples", "reconstructions",
                "interpolation", "latent_space", "denoising",
                "training_curves", "gifs"]:
        ensure_dir(os.path.join(cfg.output_dir, sub))
