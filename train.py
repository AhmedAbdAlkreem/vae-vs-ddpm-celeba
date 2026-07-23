#!/usr/bin/env python3
"""
Train the VAE then the DDPM on CelebA.

    python train.py

Adjust hyperparameters by editing configs/*.py, or by constructing a
Config(...) with overrides in your own script / notebook.
"""
from configs.config import Config
from src.engine.trainer import GenAITrainer
from src.visualization.training_curves import save_ddpm_curves, save_vae_curves


def main():
    cfg = Config()
    trainer = GenAITrainer(cfg)
    result = trainer.run()

    save_vae_curves(result["vae_history"], f"{cfg.output_dir}/training_curves/vae_curves.png")
    save_ddpm_curves(result["ddpm_history"], f"{cfg.output_dir}/training_curves/ddpm_curves.png")

    print("Done. Checkpoints saved to", cfg.checkpoint_dir)


if __name__ == "__main__":
    main()
