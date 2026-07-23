"""
Generic epoch-loop trainer shared by VAETrainer and DDPMTrainer: handles the
resume-from-checkpoint logic once, so train_vae.py / train_ddpm.py only need
to implement the actual per-batch step.
"""
import os

from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger


class BaseTrainer:
    """Subclasses implement `train_one_epoch(epoch) -> dict_of_metrics` and
    `state_dict()` / `load_state_dict()` for resumability."""

    name = "base"

    def __init__(self, cfg, total_epochs: int):
        self.cfg = cfg
        self.total_epochs = total_epochs
        self.logger = get_logger(self.name, log_dir=cfg.logs_dir)
        self.history = []
        self.start_epoch = 0

    def ckpt_path(self) -> str:
        return os.path.join(self.cfg.checkpoint_dir, f"{self.name}_checkpoint.pt")

    def final_path(self) -> str:
        return os.path.join(self.cfg.checkpoint_dir, f"{self.name}_best.pt")

    def try_resume(self, device):
        path = self.ckpt_path()
        if os.path.exists(path):
            ckpt = load_checkpoint(path, map_location=device)
            self.load_state_dict(ckpt)
            self.history = ckpt["history"]
            self.start_epoch = ckpt["epoch"] + 1
            self.logger.info(f"Resuming {self.name} training from epoch {self.start_epoch}")

    def run(self):
        if self.start_epoch >= self.total_epochs:
            self.logger.info(f"{self.name}: checkpoint already covers requested epochs, skipping.")
            return self.history

        for epoch in range(self.start_epoch, self.total_epochs):
            metrics = self.train_one_epoch(epoch)
            self.history.append(metrics)
            self.on_epoch_end(epoch, metrics)

        self.save_final()
        return self.history

    # --- subclasses must implement these ---
    def train_one_epoch(self, epoch):
        raise NotImplementedError

    def on_epoch_end(self, epoch, metrics):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, ckpt):
        raise NotImplementedError

    def save_final(self):
        raise NotImplementedError
