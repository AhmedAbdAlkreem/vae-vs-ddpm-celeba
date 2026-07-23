"""Frechet Inception Distance, via torchmetrics/torch-fidelity's InceptionV3
feature extractor (same implementation used in most published papers)."""
from src.utils.image_utils import to_uint8


def compute_fid(real, fake, device, batch=64):
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for i in range(0, len(real), batch):
        fid.update(to_uint8(real[i:i + batch]).to(device), real=True)
    for i in range(0, len(fake), batch):
        fid.update(to_uint8(fake[i:i + batch]).to(device), real=False)
    return fid.compute().item()
