"""Random seed utilities enforcing explicit randomness discipline across the lab."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def make_generator(seed: int) -> torch.Generator:
    """Create and seed an explicit torch.Generator.

    Using a dedicated Generator prevents DataLoader shuffling or sampling operations
    from mutating or depending on global torch random state.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_everything(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch global random states.

    Used at application or script entry points to establish a reproducible baseline
    for any secondary library routines that rely on global random generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_seed(seed: int) -> None:
    """Convenience alias for seed_everything, establishing explicit global seed."""
    seed_everything(seed)

