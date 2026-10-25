"""Tests for distillation experiment helpers, effective_epochs, and statistics."""

from __future__ import annotations

import math
import numpy as np
import pytest

from paradigms.distillation.experiment import effective_epochs, load_teacher
from paradigms.utils.stats import mean_std, paired_comparison


def test_effective_epochs() -> None:
    """Verify effective_epochs scaling behavior across fractions.

    1.0 -> base_epochs
    0.25 -> 80 (with base=20)
    0.01 -> 400 (with base=20, capped at 400)
    """
    base = 20
    assert effective_epochs(base, 1.0) == 20
    assert effective_epochs(base, 0.25) == 80
    assert effective_epochs(base, 0.01) == 400
    assert effective_epochs(10, 0.5) == 20

    with pytest.raises(ValueError):
        effective_epochs(base, 0.0)

    with pytest.raises(ValueError):
        effective_epochs(base, 1.5)


def test_mean_std_ddof_zero() -> None:
    """Verify mean_std uses population standard deviation with ddof=0."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, std = mean_std(values)

    assert mean == 3.0
    # Expected std = sqrt(((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5) = sqrt(2)
    assert math.isclose(std, math.sqrt(2.0), rel_tol=1e-7)
    assert std == float(np.std(values, ddof=0))

    # Empty sequence
    assert mean_std([]) == (0.0, 0.0)


def test_paired_comparison_hand_numbers() -> None:
    """Verify paired_comparison calculations against hand-computed values."""
    # a: candidate, b: baseline
    a = [0.88, 0.89, 0.90]
    b = [0.86, 0.87, 0.91]
    # diffs: [0.02, 0.02, -0.01]
    # mean_diff = 0.03 / 3 = 0.01
    # wins = 2 (0.88 > 0.86 and 0.89 > 0.87)
    # n = 3
    res = paired_comparison(a, b)

    assert math.isclose(res["mean_diff"], 0.01, rel_tol=1e-7)
    assert res["wins"] == 2
    assert res["n"] == 3
    assert math.isclose(res["std_diff"], float(np.std([0.02, 0.02, -0.01], ddof=0)), rel_tol=1e-7)

    # Mismatched lengths
    with pytest.raises(ValueError, match="Lengths must match"):
        paired_comparison([1.0], [1.0, 2.0])

    # Empty inputs
    with pytest.raises(ValueError, match="Cannot perform paired comparison"):
        paired_comparison([], [])


def test_load_teacher_freezes_and_eval_mode() -> None:
    """Verify load_teacher loads pre-trained weights, freezes parameters, and sets eval mode."""
    teacher = load_teacher(seed=0, ckpt_dir="checkpoints/teacher")
    assert teacher.training is False
    assert all(not p.requires_grad for p in teacher.parameters())
