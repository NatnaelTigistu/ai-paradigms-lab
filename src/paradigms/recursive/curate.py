"""Candidate curation selecting verified predictions subject to per-class caps."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def curate(
    indices: Sequence[int] | np.ndarray,
    preds: Sequence[int] | np.ndarray,
    confs: Sequence[float] | np.ndarray,
    verified_mask: Sequence[bool] | np.ndarray,
    cap_per_class: int,
    seed: int,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Curate verified candidates under class-wise caps using random or confidence-based selection.

    Args:
        indices: Candidate sample indices.
        preds: Model predicted class labels.
        confs: Prediction confidence scores.
        verified_mask: Boolean mask indicating verified predictions.
        cap_per_class: Maximum number of samples to retain per predicted class.
        seed: Random seed for deterministic shuffling under "random" strategy.
        strategy: Curation strategy ("random" or "confident").

    Raises:
        ValueError: If strategy is unsupported or input lengths mismatch.

    Returns:
        Tuple of (sel_idx, sel_labels) sorted by dataset index.
    """
    if strategy not in ("random", "confident"):
        raise ValueError(
            f"Unknown curation strategy '{strategy}'. Expected 'random' or 'confident'."
        )

    if len(indices) != len(preds) or len(indices) != len(confs) or len(indices) != len(verified_mask):
        raise ValueError("Input sequences must have identical lengths.")

    if len(indices) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    indices_arr = np.asarray(indices, dtype=np.int64)
    preds_arr = np.asarray(preds, dtype=np.int64)
    confs_arr = np.asarray(confs, dtype=np.float64)
    verified_arr = np.asarray(verified_mask, dtype=bool)

    # Keep verified only
    indices_v = indices_arr[verified_arr]
    preds_v = preds_arr[verified_arr]
    confs_v = confs_arr[verified_arr]

    if len(indices_v) == 0 or cap_per_class <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    selected_indices: list[int] = []
    selected_labels: list[int] = []

    unique_classes = sorted(np.unique(preds_v).tolist())

    for c in unique_classes:
        c_mask = preds_v == c
        c_idx = indices_v[c_mask]
        c_conf = confs_v[c_mask]
        count = len(c_idx)
        n_keep = min(cap_per_class, count)
        if n_keep <= 0:
            continue

        if strategy == "random":
            rng = np.random.RandomState(seed)
            perm = rng.permutation(count)
            chosen = c_idx[perm[:n_keep]]
        elif strategy == "confident":
            sort_order = np.argsort(-c_conf, kind="stable")
            chosen = c_idx[sort_order[:n_keep]]

        selected_indices.extend(chosen.tolist())
        selected_labels.extend([c] * len(chosen))

    if not selected_indices:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    sel_idx = np.asarray(selected_indices, dtype=np.int64)
    sel_labels = np.asarray(selected_labels, dtype=np.int64)
    sort_idx = np.argsort(sel_idx)

    return sel_idx[sort_idx], sel_labels[sort_idx]
