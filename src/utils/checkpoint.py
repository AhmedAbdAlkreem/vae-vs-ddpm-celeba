"""Checkpoint save/load helpers."""
import os

import torch


def save_checkpoint(path: str, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kwargs, path)


def load_checkpoint(path: str, map_location=None) -> dict:
    return torch.load(path, map_location=map_location)
