"""Light data augmentation, kept separate from the core resize/normalize
transform so it can be toggled independently (e.g. disabled during
evaluation/sampling, always disabled for reconstruction visualizations)."""
from torchvision import transforms


def build_augmentations(horizontal_flip: bool = True):
    aug = []
    if horizontal_flip:
        aug.append(transforms.RandomHorizontalFlip(p=0.5))
    return aug
