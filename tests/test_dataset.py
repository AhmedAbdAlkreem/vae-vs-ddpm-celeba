"""Sanity tests for the CelebA dataset pipeline using a synthetic temp dataset
(does not require the real CelebA download)."""
import os
import tempfile

import torch
from PIL import Image

from src.datasets.dataset import CelebADataset
from src.datasets.transforms import build_transform


def _make_fake_images(n=5, size=200):
    tmp_dir = tempfile.mkdtemp()
    for i in range(n):
        img = Image.fromarray((torch.rand(size, size, 3) * 255).byte().numpy())
        img.save(os.path.join(tmp_dir, f"{i:06d}.jpg"))
    return tmp_dir


def test_dataset_returns_correct_tensor_shape():
    tmp_dir = _make_fake_images(n=3)
    files = [os.path.join(tmp_dir, f) for f in sorted(os.listdir(tmp_dir))]
    transform = build_transform(img_size=64, augment=False)
    dataset = CelebADataset(files, transform)

    assert len(dataset) == 3
    x = dataset[0]
    assert x.shape == (3, 64, 64)
    assert x.min() >= -1.0 and x.max() <= 1.0
