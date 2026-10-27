"""Loop partition, bundle encapsulation, and leakage assertion for recursive loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from paradigms.data import FashionData
from paradigms.data.splits import stratified_subset


@dataclass(frozen=True)
class LoopPartition:
    """Immutable record of indices partitioned across recursive loop roles."""

    labeled: list[int]
    pool: list[int]
    gate: list[int]
    monitor: list[int]


def make_partition(
    data: FashionData,
    n_labeled: int,
    pool_size: int,
    seed: int,
) -> LoopPartition:
    """Create disjoint initial labeled, unlabeled pool, gate, and monitor partitions.

    Args:
        data: FashionData instance managing Fashion-MNIST splits.
        n_labeled: Target count of labeled seed training examples (~n_labeled).
        pool_size: Exact count of unlabeled pool training samples.
        seed: Random seed for deterministic stratified sampling and pool choice.

    Returns:
        LoopPartition with disjoint index sets.
    """
    train_indices = data.splits["train"]
    labels = data.train_labels
    fraction = float(n_labeled) / len(train_indices)
    labeled = stratified_subset(train_indices, labels, fraction=fraction, seed=seed)

    labeled_set = set(labeled)
    remaining = [i for i in train_indices if i not in labeled_set]
    if pool_size > len(remaining):
        raise ValueError(
            f"pool_size ({pool_size}) exceeds available remaining training samples ({len(remaining)})"
        )

    rng = np.random.RandomState(seed)
    pool = sorted(rng.choice(remaining, pool_size, replace=False).tolist())

    gate = list(data.get("gate").indices)
    monitor = list(data.get("monitor").indices)

    return LoopPartition(
        labeled=labeled,
        pool=pool,
        gate=gate,
        monitor=monitor,
    )


def assert_no_leakage(part: LoopPartition) -> None:
    """Verify that labeled, pool, gate, and monitor partitions are pairwise disjoint."""
    sets = {
        "labeled": set(part.labeled),
        "pool": set(part.pool),
        "gate": set(part.gate),
        "monitor": set(part.monitor),
    }
    names = list(sets.keys())
    overlaps: list[str] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            overlap = sets[name_a] & sets[name_b]
            if overlap:
                overlaps.append(f"{name_a} & {name_b} ({len(overlap)} overlapping indices)")

    if overlaps:
        raise AssertionError(f"Data leakage detected across recursive loop splits: {overlaps}")


@dataclass
class LoopBundle:
    """Bundle containing data datasets, hidden ground-truth labels, and index partitions."""

    base: Dataset
    hidden_labels: np.ndarray  # ONLY oracle/diagnostics may read
    partition: LoopPartition
    gate_ds: Dataset
    monitor_ds: Dataset


def make_bundle(
    data: FashionData,
    rcfg: dict[str, Any] | Any,
    seed: int,
) -> LoopBundle:
    """Create a verified LoopBundle from FashionData and recursive loop configuration.

    Args:
        data: FashionData instance.
        rcfg: Recursive loop configuration dictionary or Config object.
        seed: Execution seed.

    Returns:
        Verified LoopBundle ready for loop execution.
    """
    cfg = rcfg.get("recursive", rcfg) if hasattr(rcfg, "get") else rcfg
    n_labeled = (
        cfg.get("n_labeled", 2000)
        if hasattr(cfg, "get")
        else getattr(cfg, "n_labeled", 2000)
    )
    pool_size = (
        cfg.get("pool_size", 20000)
        if hasattr(cfg, "get")
        else getattr(cfg, "pool_size", 20000)
    )

    partition = make_partition(data, n_labeled=n_labeled, pool_size=pool_size, seed=seed)
    assert_no_leakage(partition)

    return LoopBundle(
        base=data.full_train,
        hidden_labels=data.train_labels,
        partition=partition,
        gate_ds=data.get("gate"),
        monitor_ds=data.get("monitor"),
    )
