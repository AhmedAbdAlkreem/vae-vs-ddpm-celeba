"""VAE trainer: implements BaseTrainer for the VAE model."""
import os

import torch

from src.datasets.dataloader import get_dataloader
from src.losses.vae_loss import vae_loss
from src.models.vae import VAE
from src.training.callbacks import CheckpointCallback, SampleGridCallback
from src.training.optimizer import build_vae_optimizer
from src.training.trainer import BaseTrainer
from src.utils.checkpoint import save_checkpoint
from src.utils.device import get_device
from src.utils.helpers import count_parameters, kl_weight_schedule
from src.utils.seed import set_seed


class VAETrainer(BaseTrainer):
    name = "vae"

    def __init__(self, cfg, loader=None, perceptual_loss_fn=None):
        super().__init__(cfg, total_epochs=cfg.training.vae_epochs)
        set_seed(cfg.seed)
        self.device = get_device()
        self.loader = loader if loader is not None else get_dataloader(cfg)[0]

        self.model = VAE.from_config(cfg).to(self.device)
        self.logger.info(f"VAE parameters: {count_parameters(self.model):.2f}M")
        self.opt = build_vae_optimizer(self.model, cfg)
        self.perceptual_loss_fn = perceptual_loss_fn

        self.ckpt_cb = CheckpointCallback(self.ckpt_path(), self.final_path(),
                                           every_n=cfg.training.checkpoint_every)
        self.sample_cb = SampleGridCallback(os.path.join(cfg.samples_dir, "vae"), "vae")

        self.try_resume(self.device)

    def train_one_epoch(self, epoch):
        self.model.train()
        running = {"total": 0.0, "recon": 0.0, "kld": 0.0}
        w = kl_weight_schedule(epoch, self.cfg.training.kl_warmup_epochs)

        for x in self.loader:
            x = x.to(self.device)
            recon, mu, logvar = self.model(x)
            loss, r, k, _ = vae_loss(
                recon, x, mu, logvar, kld_weight=w,
                perceptual_loss_fn=self.perceptual_loss_fn,
                perceptual_weight=self.cfg.training.perceptual_weight if self.cfg.training.use_perceptual_loss else 0.0,
            )
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            running["total"] += loss.item()
            running["recon"] += r.item()
            running["kld"] += k.item()

        n = len(self.loader)
        for key in running:
            running[key] /= n
        running["kl_w"] = w
        return running

    def on_epoch_end(self, epoch, metrics):
        self.logger.info(
            f"epoch {epoch + 1}/{self.total_epochs}  total={metrics['total']:.2f}  "
            f"recon={metrics['recon']:.2f}  kld={metrics['kld']:.2f}  kl_w={metrics['kl_w']:.2f}"
        )
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(16, self.device)
        self.sample_cb.maybe_save(epoch, self.total_epochs, samples)
        state = self.state_dict()
        state["history"] = self.history
        self.ckpt_cb.maybe_save(epoch, self.total_epochs, state)

    def state_dict(self):
        return {"model": self.model.state_dict(), "optimizer": self.opt.state_dict()}

    def load_state_dict(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])

    def save_final(self):
        save_checkpoint(self.final_path(), model=self.model.state_dict(), config=str(self.cfg))


def train_vae(cfg, loader=None, perceptual_loss_fn=None):
    trainer = VAETrainer(cfg, loader=loader, perceptual_loss_fn=perceptual_loss_fn)
    history = trainer.run()
    return trainer.model, history
