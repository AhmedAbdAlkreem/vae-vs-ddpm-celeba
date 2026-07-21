"""Image preprocessing transform pipeline."""
from torchvision import transforms

from src.datasets.augmentations import build_augmentations


def build_transform(img_size: int, augment: bool = False, horizontal_flip: bool = True) -> transforms.Compose:
    ops = [transforms.CenterCrop(178)]  # CelebA faces are roughly centered in a 178x178 box
    if augment:
        ops += build_augmentations(horizontal_flip=horizontal_flip)
    ops += [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ]
    return transforms.Compose(ops)
