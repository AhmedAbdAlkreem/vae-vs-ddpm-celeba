"""Dataset-related configuration."""
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    data_dir: str = ""          # leave empty to auto-detect via src/datasets/preprocessing.py
    img_size: int = 64
    subset_size: int = 30000
    batch_size: int = 128
    num_workers: int = 2
    horizontal_flip: bool = True   # light augmentation, see src/datasets/augmentations.py
