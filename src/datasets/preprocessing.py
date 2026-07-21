"""Locating and listing the CelebA image files on disk (Kaggle or local)."""
import glob
import os
import random

CELEBA_DIR_CANDIDATES = [
    "/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba/img_align_celeba",
    "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
    "/kaggle/input/celeba-dataset/img_align_celeba",
]


def resolve_celeba_dir(explicit_dir: str = "") -> str:
    candidates = [explicit_dir] if explicit_dir else CELEBA_DIR_CANDIDATES
    for c in candidates:
        if c and os.path.isdir(c) and len(os.listdir(c)) > 0:
            return c
    raise FileNotFoundError(
        "CelebA image folder not found in any known location. Add the "
        "'CelebFaces Attributes (CelebA) Dataset' to this Kaggle notebook, or "
        "set cfg.dataset.data_dir explicitly to the correct path."
    )


def list_image_files(data_dir: str, subset_size: int, seed: int = 42):
    img_dir = resolve_celeba_dir(data_dir)
    all_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not all_files:
        raise FileNotFoundError(f"No .jpg files found in {img_dir}")

    rng = random.Random(seed)
    rng.shuffle(all_files)
    return all_files[:subset_size]
