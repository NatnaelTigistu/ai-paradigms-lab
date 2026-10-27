"""Tests for recursive loop configuration, state partitioning, candidate generation, and curation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from paradigms.config import load_config
from paradigms.data import FashionData
from paradigms.models import SmallCNN
from paradigms.recursive import (
    Candidates,
    LoopBundle,
    LoopPartition,
    assert_no_leakage,
    curate,
    generate,
    make_bundle,
    make_partition,
)
from tests.helpers import make_synthetic


def test_recursive_config_structure() -> None:
    """Verify configs/recursive.yaml contains expected parameters."""
    cfg = load_config("configs/recursive.yaml")
    rcfg = cfg.recursive

    assert rcfg.model == "small_cnn"
    assert rcfg.model_kwargs == {}
    assert rcfg.n_labeled == 2000
    assert rcfg.pool_size == 20000
    assert rcfg.rounds == 4
    assert rcfg.epochs_initial == 30
    assert rcfg.epochs_per_round == 4
    assert rcfg.lr_round == 0.0005
    assert rcfg.batch_size == 128
    assert rcfg.cap_per_class == 300
    assert rcfg.curation == "random"
    assert rcfg.agreement_threshold == 0.9
    assert rcfg.self_conf_threshold == 0.9
    assert rcfg.gate.enabled is True
    assert rcfg.gate.min_improvement == 0.001
    assert rcfg.gate.max_class_drop == 0.05
    assert rcfg.gate.patience == 2
    assert list(rcfg.seeds) == [0, 1, 2]


def test_partition_counts_leakage_and_determinism() -> None:
    """Verify partition counts (~2000 labeled, 20000 pool), no leakage, and seed behavior."""
    data = FashionData(root="./data", split_seed=42)

    p0 = make_partition(data, n_labeled=2000, pool_size=20000, seed=0)
    p0_dup = make_partition(data, n_labeled=2000, pool_size=20000, seed=0)
    p1 = make_partition(data, n_labeled=2000, pool_size=20000, seed=1)

    # Counts
    assert abs(len(p0.labeled) - 2000) <= 10
    assert len(p0.pool) == 20000
    assert len(p0.gate) + len(p0.monitor) == len(data.get("dev"))

    # No leakage
    assert_no_leakage(p0)
    assert_no_leakage(p1)

    # Deterministic per seed
    assert p0.labeled == p0_dup.labeled
    assert p0.pool == p0_dup.pool
    assert p0.gate == p0_dup.gate
    assert p0.monitor == p0_dup.monitor

    # Different seed gives different labeled and pool sets, but identical gate and monitor
    assert p0.labeled != p1.labeled
    assert p0.pool != p1.pool
    assert p0.gate == p1.gate
    assert p0.monitor == p1.monitor


def test_assert_no_leakage_raises_on_overlap() -> None:
    """Verify assert_no_leakage detects any pairwise overlap between loop partitions."""
    clean = LoopPartition(
        labeled=[1, 2, 3],
        pool=[4, 5, 6],
        gate=[7, 8],
        monitor=[9, 10],
    )
    assert_no_leakage(clean)

    # Leaked between labeled and pool
    leaked_pool = LoopPartition(
        labeled=[1, 2, 3],
        pool=[3, 4, 5],
        gate=[7, 8],
        monitor=[9, 10],
    )
    with pytest.raises(AssertionError, match="Data leakage"):
        assert_no_leakage(leaked_pool)

    # Leaked between gate and monitor
    leaked_gate = LoopPartition(
        labeled=[1, 2, 3],
        pool=[4, 5, 6],
        gate=[7, 8],
        monitor=[8, 9],
    )
    with pytest.raises(AssertionError, match="Data leakage"):
        assert_no_leakage(leaked_gate)


def test_make_bundle() -> None:
    """Verify make_bundle returns properly wired LoopBundle."""
    data = FashionData(root="./data", split_seed=42)
    cfg = load_config("configs/recursive.yaml")

    bundle = make_bundle(data, cfg, seed=0)
    assert isinstance(bundle, LoopBundle)
    assert bundle.base is data.full_train
    assert np.array_equal(bundle.hidden_labels, data.train_labels)
    assert isinstance(bundle.partition, LoopPartition)
    assert len(bundle.partition.pool) == 20000
    assert len(bundle.gate_ds) == len(bundle.partition.gate)
    assert len(bundle.monitor_ds) == len(bundle.partition.monitor)


def test_generate_preserves_order() -> None:
    """Verify generate preserves index order and matches individual sample predictions."""
    model = SmallCNN()
    model.eval()

    # Synthetic dataset with 20 items
    base = make_synthetic(30, seed=99)
    indices = [17, 3, 25, 0, 11]

    cands = generate(model, base, indices, batch_size=2)
    assert isinstance(cands, Candidates)
    assert np.array_equal(cands.indices, np.array(indices))
    assert len(cands.preds) == len(indices)
    assert len(cands.confs) == len(indices)

    # Verify per-sample prediction comparison
    with torch.no_grad():
        for i, idx in enumerate(indices):
            x, _ = base[idx]
            logits = model(x.unsqueeze(0))
            prob = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(prob, dim=-1)

            assert cands.preds[i] == pred.item()
            assert abs(cands.confs[i] - conf.item()) < 1e-5

    # Empty indices test
    empty_cands = generate(model, base, [], batch_size=2)
    assert len(empty_cands.indices) == 0
    assert len(empty_cands.preds) == 0
    assert len(empty_cands.confs) == 0


def test_curate_caps_verification_and_strategies() -> None:
    """Verify curate filters non-verified, respects per-class caps, and implements strategies."""
    indices = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    preds = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    confs = np.array([0.60, 0.95, 0.80, 0.70, 0.90, 0.85, 0.50, 0.99])
    verified = np.array([True, True, True, True, True, False, True, False])

    # Class 0: verified idx [10, 20, 30], confs [0.60, 0.95, 0.80]
    # Class 1: verified idx [40, 50] (idx 60 not verified), confs [0.70, 0.90]
    # Class 2: verified idx [70] (idx 80 not verified), confs [0.50]

    # Test "confident" with cap_per_class=2
    # Class 0: top 2 are idx 20 (0.95) and idx 30 (0.80)
    # Class 1: top 2 are idx 50 (0.90) and idx 40 (0.70)
    # Class 2: only 1 verified, idx 70 (0.50)
    # Sorted indices should be: [20, 30, 40, 50, 70]
    sel_idx, sel_labels = curate(
        indices, preds, confs, verified, cap_per_class=2, seed=42, strategy="confident"
    )

    assert np.array_equal(sel_idx, np.array([20, 30, 40, 50, 70]))
    assert np.array_equal(sel_labels, np.array([0, 0, 1, 1, 2]))

    # Test "random" strategy determinism
    r_idx1, r_labels1 = curate(
        indices, preds, confs, verified, cap_per_class=1, seed=123, strategy="random"
    )
    r_idx2, r_labels2 = curate(
        indices, preds, confs, verified, cap_per_class=1, seed=123, strategy="random"
    )
    assert np.array_equal(r_idx1, r_idx2)
    assert np.array_equal(r_labels1, r_labels2)
    # 1 item per class, exactly 3 classes
    assert len(r_idx1) == 3

    # Test empty inputs
    empty_idx, empty_labels = curate([], [], [], [], cap_per_class=2, seed=0, strategy="random")
    assert len(empty_idx) == 0
    assert len(empty_labels) == 0

    # Test all unverified
    all_false = np.zeros(len(indices), dtype=bool)
    none_idx, none_labels = curate(
        indices, preds, confs, all_false, cap_per_class=2, seed=0, strategy="confident"
    )
    assert len(none_idx) == 0
    assert len(none_labels) == 0
