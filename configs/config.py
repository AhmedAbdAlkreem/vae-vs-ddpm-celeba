"""
Top-level configuration composing dataset / model / diffusion / training
sub-configs. Import `Config` and override any field, e.g.:

    from configs.config import Config
    cfg = Config()
    cfg.training.vae_epochs = 50
    cfg.dataset.subset_size = 50000
"""
from dataclasses import dataclass, field

from configs.dataset import DatasetConfig
from configs.model import ModelConfig
from configs.diffusion import DiffusionConfig
from configs.training import TrainingConfig


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "outputs"

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Evaluation
    n_eval_samples: int = 2000
    eval_batch_size: int = 64

    # --- convenience path properties (kept in sync with output_dir) ---
    @property
    def checkpoint_dir(self) -> str:
        return f"{self.output_dir}/checkpoints"

    @property
    def samples_dir(self) -> str:
        return f"{self.output_dir}/samples"

    @property
    def logs_dir(self) -> str:
        return f"{self.output_dir}/logs"
