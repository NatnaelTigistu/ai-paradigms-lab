"""Tests for recursive loop verification mechanisms and structural security."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from paradigms.recursive import (
    AgreementVerifier,
    LoopBundle,
    LoopPartition,
    NoVerifier,
    OracleVerifier,
    SelfConfidenceVerifier,
    build_verifier,
)


def test_oracle_verifier() -> None:
    """Verify OracleVerifier accepts exactly where predictions match hidden ground truth."""
    hidden = np.array([0, 1, 2, 3, 4, 5])
    oracle = OracleVerifier(hidden_labels=hidden)

    indices = [0, 1, 2, 3, 4]
    preds = [0, 9, 2, 8, 4]  # Matches at 0, 2, 4
    confs = [0.1, 0.99, 0.5, 0.99, 0.2]

    verified = oracle.verify(indices, preds, confs)
    expected = np.array([True, False, True, False, True])
    assert np.array_equal(verified, expected)

    # Empty indices
    empty_verified = oracle.verify([], [], [])
    assert len(empty_verified) == 0


def test_agreement_verifier_precision_on_separable_data() -> None:
    """Verify agreement precision of accepted > accuracy of all predictions on separable data."""
    rng = np.random.RandomState(42)

    # Linearly separable 2-class synthetic data: class 0 around -5.0, class 1 around +5.0
    n_labeled_per_class = 60
    x_labeled_c0 = rng.normal(-5.0, 0.5, size=(n_labeled_per_class, 10)).astype(np.float32)
    x_labeled_c1 = rng.normal(5.0, 0.5, size=(n_labeled_per_class, 10)).astype(np.float32)
    labeled_images = np.vstack([x_labeled_c0, x_labeled_c1])
    labeled_y = np.array([0] * n_labeled_per_class + [1] * n_labeled_per_class, dtype=np.int64)

    # Pool set: 100 samples
    n_pool_per_class = 50
    x_pool_c0 = rng.normal(-5.0, 0.5, size=(n_pool_per_class, 10)).astype(np.float32)
    x_pool_c1 = rng.normal(5.0, 0.5, size=(n_pool_per_class, 10)).astype(np.float32)
    pool_images = np.vstack([x_pool_c0, x_pool_c1])
    pool_true_y = np.array([0] * n_pool_per_class + [1] * n_pool_per_class, dtype=np.int64)
    pool_indices = np.arange(100, dtype=np.int64)

    # Generator model predictions: 65% accuracy (65 correct, 35 flipped), high confidence 0.95
    preds = pool_true_y.copy()
    noisy_indices = rng.choice(100, size=35, replace=False)
    preds[noisy_indices] = 1 - preds[noisy_indices]
    confs = np.full(100, 0.95, dtype=np.float64)

    raw_accuracy = (preds == pool_true_y).mean()
    assert abs(raw_accuracy - 0.65) < 1e-6

    # Fit AgreementVerifier
    verifier = AgreementVerifier(threshold=0.8, seed=42)
    verifier.fit(
        labeled_images=labeled_images,
        labeled_y=labeled_y,
        pool_indices=pool_indices,
        pool_images=pool_images,
    )

    verified_mask = verifier.verify(pool_indices, preds, confs)
    accepted_count = verified_mask.sum()
    assert accepted_count > 0

    accepted_precision = (preds[verified_mask] == pool_true_y[verified_mask]).mean()

    # Precision of accepted items must exceed baseline prediction accuracy
    assert accepted_precision > raw_accuracy
    assert accepted_precision >= 0.95


def test_self_confidence_boundary() -> None:
    """Verify self-confidence verifier boundary is strictly >=."""
    verifier = SelfConfidenceVerifier(threshold=0.85)

    indices = [0, 1, 2]
    preds = [0, 1, 2]
    confs = [0.8499, 0.8500, 0.8501]

    verified = verifier.verify(indices, preds, confs)
    assert np.array_equal(verified, np.array([False, True, True]))


def test_no_verifier_accepts_nothing() -> None:
    """Verify NoVerifier always rejects all proposed candidates."""
    verifier = NoVerifier()
    indices = [1, 2, 3]
    preds = [0, 1, 2]
    confs = [1.0, 1.0, 1.0]

    verified = verifier.verify(indices, preds, confs)
    assert not verified.any()
    assert len(verified) == 3


def test_signature_structural_enforcement() -> None:
    """Verify inspect.signature shows ONLY OracleVerifier takes hidden labels in __init__."""
    classes = [OracleVerifier, AgreementVerifier, SelfConfidenceVerifier, NoVerifier]

    for cls in classes:
        sig = inspect.signature(cls.__init__)
        if cls is OracleVerifier:
            assert "hidden_labels" in sig.parameters, "OracleVerifier must accept hidden_labels"
        else:
            assert "hidden_labels" not in sig.parameters, (
                f"{cls.__name__} must NOT accept hidden_labels in __init__"
            )


def test_non_oracle_verifiers_immune_to_shuffled_hidden_labels() -> None:
    """Verify building non-oracle verifiers on a bundle with shuffled hidden labels gives identical outputs."""
    rng = np.random.RandomState(42)

    # Create synthetic base dataset with .data (uint8) and .targets (tensor)
    n_total = 200
    images_raw = torch.randint(0, 256, (n_total, 28, 28), dtype=torch.uint8)
    true_labels = torch.randint(0, 10, (n_total,), dtype=torch.long)

    class SyntheticBase:
        def __init__(self, data: torch.Tensor, targets: torch.Tensor) -> None:
            self.data = data
            self.targets = targets

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            return self.data[idx], int(self.targets[idx])

    base = SyntheticBase(images_raw, true_labels)

    labeled_idx = list(range(0, 50))
    pool_idx = list(range(50, 150))
    gate_idx = list(range(150, 175))
    monitor_idx = list(range(175, 200))

    partition = LoopPartition(
        labeled=labeled_idx,
        pool=pool_idx,
        gate=gate_idx,
        monitor=monitor_idx,
    )

    clean_hidden = true_labels.numpy().copy()
    bundle_clean = LoopBundle(
        base=base,
        hidden_labels=clean_hidden,
        partition=partition,
        gate_ds=TensorDataset(torch.empty(0), torch.empty(0)),
        monitor_ds=TensorDataset(torch.empty(0), torch.empty(0)),
    )

    # Shuffled hidden labels bundle
    shuffled_hidden = rng.permutation(clean_hidden)
    assert not np.array_equal(clean_hidden, shuffled_hidden)

    bundle_shuffled = LoopBundle(
        base=base,
        hidden_labels=shuffled_hidden,
        partition=partition,
        gate_ds=TensorDataset(torch.empty(0), torch.empty(0)),
        monitor_ds=TensorDataset(torch.empty(0), torch.empty(0)),
    )

    rcfg = {
        "recursive": {
            "agreement_threshold": 0.8,
            "self_conf_threshold": 0.8,
        }
    }

    test_preds = rng.randint(0, 10, size=len(pool_idx))
    test_confs = rng.uniform(0.5, 1.0, size=len(pool_idx))

    # 1. Non-oracle verifiers must produce IDENTICAL results
    for vname in ["agreement", "self_confidence", "none"]:
        v_clean = build_verifier(vname, rcfg, bundle_clean, seed=42)
        v_shuffled = build_verifier(vname, rcfg, bundle_shuffled, seed=42)

        out_clean = v_clean.verify(pool_idx, test_preds, test_confs)
        out_shuffled = v_shuffled.verify(pool_idx, test_preds, test_confs)

        assert np.array_equal(out_clean, out_shuffled), (
            f"Verifier '{vname}' produced different outputs when hidden_labels were shuffled!"
        )

    # 2. OracleVerifier MUST produce DIFFERENT results when hidden_labels are shuffled
    v_oracle_clean = build_verifier("oracle", rcfg, bundle_clean, seed=42)
    v_oracle_shuffled = build_verifier("oracle", rcfg, bundle_shuffled, seed=42)

    out_oracle_clean = v_oracle_clean.verify(pool_idx, test_preds, test_confs)
    out_oracle_shuffled = v_oracle_shuffled.verify(pool_idx, test_preds, test_confs)

    assert not np.array_equal(out_oracle_clean, out_oracle_shuffled), (
        "OracleVerifier output should change when hidden_labels are shuffled"
    )
