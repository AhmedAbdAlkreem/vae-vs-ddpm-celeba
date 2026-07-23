"""DDPM trainer: implements BaseTrainer for the U-Net + GaussianDiffusion."""
import os

import torch

from src.datasets.dataloader import get_dataloader
from src.models.diffusion import GaussianDiffusion
from src.models.unet import UNet
from src.sampling.ddpm_sampler import ddpm_sample
from src.training.callbacks import CheckpointCallback, SampleGridCallback
from src.training.ema import EMA
from src.training.optimizer import build_ddpm_optimizer
from src.training.trainer import BaseTrainer
from src.utils.checkpoint import save_checkpoint
from src.utils.device import get_device
from src.utils.helpers import count_parameters
from src.utils.seed import set_seed


class DDPMTrainer(BaseTrainer):
    name = "ddpm"

    def __init__(self, cfg, loader=None):
        super().__init__(cfg, total_epochs=cfg.training.ddpm_epochs)
        set_seed(cfg.seed)
        self.device = get_device()
        self.loader = loader if loader is not None else get_dataloader(cfg)[0]

        self.model = UNet.from_config(cfg).to(self.device)
        self.logger.info(f"UNet parameters: {count_parameters(self.model):.2f}M")
        self.diffusion = GaussianDiffusion.from_config(cfg, self.device)

        self.ema = EMA(self.model, decay=cfg.diffusion.ema_decay)
        self.opt = build_ddpm_optimizer(self.model, cfg)
        self.use_amp = cfg.training.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.ckpt_cb = CheckpointCallback(self.ckpt_path(), self.final_path(),
                                           every_n=cfg.training.checkpoint_every)
        self.sample_cb = SampleGridCallback(os.path.join(cfg.samples_dir, "ddpm"), "ddpm")

        self.try_resume(self.device)

    def train_one_epoch(self, epoch):
        self.model.train()
        running = 0.0
        for x in self.loader:
            x = x.to(self.device)
            self.opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss = self.diffusion.training_loss(self.model, x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            self.ema.update(self.model)
            running += loss.item()
        return {"loss": running / len(self.loader)}

    def on_epoch_end(self, epoch, metrics):
        self.logger.info(f"epoch {epoch + 1}/{self.total_epochs}  loss={metrics['loss']:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == self.total_epochs - 1:
            eval_unet = UNet.from_config(self.cfg).to(self.device)
            self.ema.copy_to(eval_unet)
            eval_unet.eval()
            samples = ddpm_sample(eval_unet, self.diffusion, self.cfg.dataset.img_size,
                                   batch_size=16, device=self.device)
            self.sample_cb.maybe_save(epoch, self.total_epochs, samples)

        self.ckpt_cb.maybe_save(epoch, self.total_epochs, {**self.state_dict(), "history": self.history})

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.ema.load_state_dict(ckpt["ema"])

    def save_final(self):
        save_checkpoint(self.final_path(), model=self.model.state_dict(),
                         ema=self.ema.state_dict(), config=str(self.cfg))


def train_ddpm(cfg, loader=None):
    trainer = DDPMTrainer(cfg, loader=loader)
    history = trainer.run()
    return trainer.model, trainer.diffusion, trainer.ema, history
