"""Dataset split utilities and stratified nested subset selection.

Fashion-MNIST comprises 60,000 training images and 10,000 official test images.
The 60,000 samples are partitioned into 50,000 "train" (for model fitting) and
10,000 "dev" (for hyperparameter tuning and model selection). The 10,000 official
test images serve strictly as the final evaluation set and are guarded against
premature access.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from paradigms.utils.io import load_json, save_json


def make_splits(n_total: int, dev_size: int, split_seed: int) -> dict[str, list[int]]:
    """Partition `n_total` indices into disjoint train and dev index sets.

    Uses an explicit RandomState seed to guarantee deterministic, reproducible
    partitions across machines and runs.

    Args:
        n_total: Total number of samples in the raw training pool (e.g. 60,000).
        dev_size: Number of samples reserved for the development set (e.g. 10,000).
        split_seed: Explicit seed for random permutation.

    Returns:
        Dictionary with sorted index lists: `{"dev": [...], "train": [...]}`.
    """
    if dev_size < 0 or dev_size > n_total:
        raise ValueError(f"dev_size must be between 0 and {n_total}, got {dev_size}")

    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(n_total)
    return {
        "dev": sorted(perm[:dev_size].tolist()),
        "train": sorted(perm[dev_size:].tolist()),
    }


def save_splits(splits: dict[str, list[int]], path: str | Path) -> None:
    """Serialize split dictionary to a JSON file on disk."""
    save_json(splits, path)


def load_splits(path: str | Path) -> dict[str, list[int]]:
    """Load precomputed split dictionary from a JSON file."""
    return load_json(path)


def stratified_subset(
    indices: Sequence[int],
    labels: np.ndarray | Sequence[int],
    fraction: float,
    seed: int,
) -> list[int]:
    """Select a stratified, nested subset of `indices` preserving class balance.

    Why nested subsets:
    By fixing the random shuffle order for each class using an explicit seed,
    any smaller subset (e.g., 5% data fraction) is guaranteed to be an exact
    subset of any larger subset (e.g., 10% data fraction). This removes sampling
    variance across data-fraction sweeps, ensuring observed performance trends
    reflect actual sample efficiency rather than lucky/unlucky data draws.

    Args:
        indices: Base pool of sample indices (e.g. train indices).
        labels: Full array of dataset targets indexed by dataset index.
        fraction: Proportion of samples to retain in (0.0, 1.0].
        seed: Explicit seed used to shuffle class indices deterministically.

    Returns:
        Sorted list of chosen dataset indices.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0.0, 1.0], got {fraction}")

    if fraction == 1.0:
        return sorted(indices)

    # Group available indices by class
    unique_classes = sorted({int(labels[i]) for i in indices})
    selected: list[int] = []

    for c in unique_classes:
        idx_c = [i for i in indices if int(labels[i]) == c]
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(idx_c).tolist()
        n_keep = math.ceil(fraction * len(idx_c))
        selected.extend(shuffled[:n_keep])

    return sorted(selected)


def split_dev(
    dev_indices: Sequence[int],
    labels: np.ndarray | Sequence[int],
    seed: int,
) -> dict[str, list[int]]:
    """Partition development indices into balanced gate and monitor sets per class."""
    unique_classes = sorted({int(labels[i]) for i in dev_indices})
    gate_indices: list[int] = []
    monitor_indices: list[int] = []

    for c in unique_classes:
        idx_c = [i for i in dev_indices if int(labels[i]) == c]
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(idx_c).tolist()
        half = len(shuffled) // 2
        gate_indices.extend(shuffled[:half])
        monitor_indices.extend(shuffled[half:])

    return {
        "gate": sorted(gate_indices),
        "monitor": sorted(monitor_indices),
    }

