"""Statistical utilities for multi-seed aggregation and paired model comparisons."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Calculate sample mean and population standard deviation (ddof=0).

    Args:
        values: Sequence of numeric values (e.g., metric across seeds).

    Returns:
        Tuple of (mean, std) as floats.
    """
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def paired_comparison(a: Sequence[float], b: Sequence[float]) -> dict[str, float | int]:
    """Compute paired differences between two matched seed series (e.g. distilled vs hard).

    Args:
        a: Metric sequence for candidate model across ordered seeds.
        b: Metric sequence for baseline model across the identical seeds.

    Raises:
        ValueError: If sequence lengths do not match or are empty.

    Returns:
        Dictionary containing:
            - "mean_diff": Mean of (a_i - b_i).
            - "std_diff": Population standard deviation of (a_i - b_i) with ddof=0.
            - "wins": Number of seeds where a_i > b_i.
            - "n": Number of paired samples.
    """
    if len(a) != len(b):
        raise ValueError(f"Lengths must match for paired comparison: len(a)={len(a)}, len(b)={len(b)}")
    if len(a) == 0:
        raise ValueError("Cannot perform paired comparison on empty sequences")

    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    wins = int(np.sum(diffs > 0))
    mean_d, std_d = mean_std(diffs)

    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "wins": wins,
        "n": len(a),
    }
