"""Unit and regression tests for Phase 1 v2 label-efficient distillation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from paradigms.data.fashion_mnist import FashionData, MEAN, STD
from paradigms.data.tensor_loader import TensorLoader, make_fast_loader
from paradigms.distillation.cached import cache_teacher_logits, masked_distillation_loss
from paradigms.distillation.experiment import load_teacher
from paradigms.distillation.loss import distillation_loss_from_logits
from paradigms.distillation.v2 import run_student_v2


def test_masked_loss_equals_v1_when_all_labelled():
    """Verify masked loss matches distillation_loss_from_logits when all labels >= 0."""
    torch.manual_seed(42)
    s_logits = torch.randn(16, 10, requires_grad=True)
    t_logits = torch.randn(16, 10)
    labels = torch.randint(0, 10, (16,))
    T = 3.0
    alpha = 0.4

    loss_v1 = distillation_loss_from_logits(s_logits, t_logits, labels, T=T, alpha=alpha)
    loss_masked = masked_distillation_loss(s_logits, labels, t_logits, T=T, alpha=alpha)

    assert torch.allclose(loss_v1, loss_masked, atol=1e-6)

    # Gradient match
    g1 = torch.autograd.grad(loss_v1, s_logits, retain_graph=True)[0]
    g2 = torch.autograd.grad(loss_masked, s_logits, retain_graph=True)[0]
    assert torch.allclose(g1, g2, atol=1e-6)


def test_unlabelled_samples_contribute_only_kd():
    """Verify unlabelled samples (label == -1) contribute pure KD regardless of alpha."""
    torch.manual_seed(42)
    s_logits = torch.randn(16, 10)
    t_logits = torch.randn(16, 10)
    labels_unlabelled = torch.full((16,), -1, dtype=torch.long)
    T = 2.0

    pure_kd = masked_distillation_loss(s_logits, labels_unlabelled, t_logits, T=T, alpha=0.0)

    for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
        loss = masked_distillation_loss(s_logits, labels_unlabelled, t_logits, T=T, alpha=alpha)
        assert torch.allclose(loss, pure_kd, atol=1e-6), f"Mismatch at alpha={alpha}"


def test_tensor_loader_covers_each_sample_once_per_epoch():
    """Verify TensorLoader covers every sample exactly once per epoch and advances seed."""
    N = 105
    X = torch.arange(N).unsqueeze(1).float()
    Y = torch.arange(N).long()

    loader = TensorLoader(X, Y, batch_size=16, shuffle=True, seed=999)
    assert len(loader) == 7  # ceil(105 / 16) = 7

    # Epoch 1
    seen1 = torch.cat([y for _, y in loader])
    assert len(seen1) == N
    assert torch.equal(seen1.sort().values, torch.arange(N))

    # Epoch 2 (shuffling advances)
    seen2 = torch.cat([y for _, y in loader])
    assert len(seen2) == N
    assert torch.equal(seen2.sort().values, torch.arange(N))
    assert not torch.equal(seen1, seen2), "Generator should advance and produce different permutation across epochs"


def test_cached_logits_equal_teacher_forward():
    """Verify cached logits match direct teacher forward pass on 100 random images within 1e-5."""
    data = FashionData("./data")
    teacher = load_teacher(0)
    cache = cache_teacher_logits(teacher, data, path="checkpoints/v2/teacher_logits.pt")

    assert cache.shape == (60000, 10)

    # 100 deterministic random indices
    rng = torch.Generator().manual_seed(12345)
    indices = torch.randint(0, 60000, (100,), generator=rng)

    raw_x = data.full_train.data[indices]
    x = ((raw_x.float() / 255.0 - MEAN) / STD).unsqueeze(1)
    with torch.no_grad():
        direct = teacher(x)

    diff = (cache[indices] - direct).abs().max().item()
    assert diff < 1e-5, f"Cache discrepancy too high: {diff}"


def test_alpha_1_distilled_equals_hard_run():
    """Verify alpha=1.0 distilled run is identical to the hard run with the same seed."""
    data = FashionData("./data")
    teacher = load_teacher(0)
    cache = cache_teacher_logits(teacher, data, path="checkpoints/v2/teacher_logits.pt")
    cfg = {"data": {"split_seed": 42}}

    hard_out = run_student_v2(
        cfg=cfg,
        data=data,
        cache=cache,
        seed=77,
        mode="hard",
        fraction=0.01,
        base_epochs=1,
    )

    dist_out = run_student_v2(
        cfg=cfg,
        data=data,
        cache=cache,
        seed=77,
        mode="distilled",
        fraction=0.01,
        T=1.0,
        alpha=1.0,
        base_epochs=1,
    )

    assert hard_out["dev_accuracy"] == dist_out["dev_accuracy"]
    assert pytest.approx(hard_out["history"][0]["train_loss"], abs=1e-5) == dist_out["history"][0]["train_loss"]
    assert pytest.approx(hard_out["history"][0]["dev_loss"], abs=1e-5) == dist_out["history"][0]["dev_loss"]


def test_transfer_mode_never_reads_unlabelled_labels():
    """Verify transfer mode results are completely invariant to the true labels of unlabelled images."""
    data = FashionData("./data")
    teacher = load_teacher(0)
    cache = cache_teacher_logits(teacher, data, path="checkpoints/v2/teacher_logits.pt")
    cfg = {"data": {"split_seed": 42}}

    # 1. Normal transfer run
    res_normal = run_student_v2(
        cfg=cfg,
        data=data,
        cache=cache,
        seed=101,
        mode="distilled_transfer",
        fraction=0.01,
        T=2.0,
        alpha=0.5,
        base_epochs=1,
    )

    # 2. Corrupt unlabelled train sample targets in data.full_train.targets
    orig_targets = data.full_train.targets.clone()
    subset = data.train_indices_subset(0.01, seed=42)
    subset_set = set(subset.indices)
    unlabelled_train = set(data.splits["train"]) - subset_set

    try:
        # Fill unlabelled train samples with bogus targets (999)
        corrupted_targets = orig_targets.clone()
        for idx in unlabelled_train:
            corrupted_targets[idx] = 999
        data.full_train.targets = corrupted_targets

        res_corrupted = run_student_v2(
            cfg=cfg,
            data=data,
            cache=cache,
            seed=101,
            mode="distilled_transfer",
            fraction=0.01,
            T=2.0,
            alpha=0.5,
            base_epochs=1,
        )

        assert res_normal["dev_accuracy"] == res_corrupted["dev_accuracy"]
        assert pytest.approx(res_normal["history"][0]["train_loss"], abs=1e-5) == res_corrupted["history"][0]["train_loss"]
    finally:
        # Restore original targets
        data.full_train.targets = orig_targets
