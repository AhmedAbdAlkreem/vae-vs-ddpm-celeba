"""Inception Score. Noted limitation: IS is derived from an ImageNet-1000-class
classifier and is less informative for a single-class domain like faces —
report it, but weight FID more heavily in conclusions (see README)."""
from src.utils.image_utils import to_uint8


def compute_inception_score(fake, device, batch=64):
    from torchmetrics.image.inception import InceptionScore

    inc = InceptionScore(normalize=False).to(device)
    for i in range(0, len(fake), batch):
        inc.update(to_uint8(fake[i:i + batch]).to(device))
    is_mean, is_std = inc.compute()
    return is_mean.item(), is_std.item()
