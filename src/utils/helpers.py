"""Small stateless helpers used across modules."""
import torch


def kl_weight_schedule(epoch: int, warmup_epochs: int, max_w: float = 1.0) -> float:
    """Linear KL-annealing schedule: 0 -> max_w over `warmup_epochs` epochs."""
    return max_w * min(1.0, epoch / max(1, warmup_epochs))


def count_parameters(model: torch.nn.Module) -> float:
    """Returns parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
