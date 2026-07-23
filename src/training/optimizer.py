"""Optimizer construction, centralized so trainers don't hardcode Adam vs AdamW."""
import torch


def build_vae_optimizer(model, cfg):
    return torch.optim.Adam(model.parameters(), lr=cfg.training.vae_lr)


def build_ddpm_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.training.ddpm_lr)
