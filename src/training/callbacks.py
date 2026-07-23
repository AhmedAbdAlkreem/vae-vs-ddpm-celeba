"""Lightweight training callbacks: checkpointing and periodic sample previews.
Kept as plain classes (not a framework) — call `.on_epoch_end(...)` manually
from the trainer loop.
"""
import os

from src.utils.checkpoint import save_checkpoint
from src.utils.image_utils import to_grid
from src.visualization.training_curves import save_curve_plot


class CheckpointCallback:
    """Saves a resumable checkpoint every `every_n` epochs, and always on the
    final epoch, plus a lightweight final-weights-only file."""

    def __init__(self, ckpt_path: str, final_path: str, every_n: int = 2):
        self.ckpt_path = ckpt_path
        self.final_path = final_path
        self.every_n = every_n

    def maybe_save(self, epoch: int, total_epochs: int, state: dict):
        if (epoch + 1) % self.every_n == 0 or epoch == total_epochs - 1:
            save_checkpoint(self.ckpt_path, epoch=epoch, **state)
            return True
        return False

    def save_final(self, **state):
        save_checkpoint(self.final_path, **state)


class SampleGridCallback:
    """Saves a sample grid image every `every_n` epochs (in addition to any
    inline plt.show() the trainer does)."""

    def __init__(self, out_dir: str, prefix: str, every_n: int = 5):
        self.out_dir = out_dir
        self.prefix = prefix
        self.every_n = every_n
        os.makedirs(out_dir, exist_ok=True)

    def maybe_save(self, epoch: int, total_epochs: int, samples):
        import matplotlib.pyplot as plt

        if (epoch + 1) % self.every_n == 0 or epoch == total_epochs - 1:
            grid = to_grid(samples, nrow=4)
            path = os.path.join(self.out_dir, f"{self.prefix}_epoch{epoch + 1:03d}.png")
            plt.figure(figsize=(5, 5))
            plt.axis("off")
            plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            return path
        return None
