"""Builds the CelebA DataLoader from a Config."""
import torch
from torch.utils.data import DataLoader

from src.datasets.dataset import CelebADataset
from src.datasets.preprocessing import list_image_files
from src.datasets.transforms import build_transform


def get_dataloader(cfg, augment: bool = True):
    """Returns (dataloader, list_of_file_paths, eval_transform).

    `eval_transform` has no augmentation applied — use it for evaluation /
    reconstruction / sampling comparisons where you want deterministic
    preprocessing of real images.
    """
    files = list_image_files(cfg.dataset.data_dir, cfg.dataset.subset_size, seed=cfg.seed)

    train_transform = build_transform(cfg.dataset.img_size, augment=augment,
                                       horizontal_flip=cfg.dataset.horizontal_flip)
    eval_transform = build_transform(cfg.dataset.img_size, augment=False)

    dataset = CelebADataset(files, train_transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, files, eval_transform
