"""Acceptance gate decision logic for recursive loop candidates."""

from __future__ import annotations

from typing import Any


def decide_gate(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    gate_cfg: dict[str, Any],
) -> tuple[bool, str]:
    """Decide whether to accept a recursive candidate model over the baseline.

    A candidate is accepted only if:
    1. Overall gate accuracy improves by at least `min_improvement`.
    2. No individual class accuracy drops by more than `max_class_drop` relative
       to baseline (for classes present in both evaluations).

    Args:
        candidate: Evaluation metrics dict for candidate model on gate set.
        baseline: Evaluation metrics dict for baseline model on gate set.
        gate_cfg: Configuration dictionary with keys:
            - min_improvement: Minimum overall accuracy gain required (float).
            - max_class_drop: Maximum allowable per-class accuracy drop (float).

    Returns:
        Tuple of (accepted: bool, reason: str).
    """
    min_improvement = (
        gate_cfg.get("min_improvement", 0.001)
        if hasattr(gate_cfg, "get")
        else getattr(gate_cfg, "min_improvement", 0.001)
    )
    max_class_drop = (
        gate_cfg.get("max_class_drop", 0.05)
        if hasattr(gate_cfg, "get")
        else getattr(gate_cfg, "max_class_drop", 0.05)
    )

    cand_acc = (
        candidate["accuracy"]
        if isinstance(candidate, dict)
        else getattr(candidate, "accuracy")
    )
    base_acc = (
        baseline["accuracy"]
        if isinstance(baseline, dict)
        else getattr(baseline, "accuracy")
    )

    gain = cand_acc - base_acc
    if gain < min_improvement:
        return False, f"rejected: gain {gain:+.4f} below {min_improvement}"

    cand_pc = (
        candidate.get("per_class_accuracy", candidate.get("pc", []))
        if isinstance(candidate, dict)
        else getattr(candidate, "per_class_accuracy", getattr(candidate, "pc", []))
    )
    base_pc = (
        baseline.get("per_class_accuracy", baseline.get("pc", []))
        if isinstance(baseline, dict)
        else getattr(baseline, "per_class_accuracy", getattr(baseline, "pc", []))
    )

    num_classes = min(len(cand_pc), len(base_pc))
    for c in range(num_classes):
        cand_val = cand_pc[c]
        base_val = base_pc[c]
        if cand_val is not None and base_val is not None:
            drop = base_val - cand_val
            if cand_val < base_val - max_class_drop:
                return False, f"rejected: class {c} dropped {drop:.3f}"

    return True, "accepted"
