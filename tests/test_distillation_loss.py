"""Tests for distillation loss formulation, factory interface, and equivalence."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from paradigms.data.fashion_mnist import make_loader
from paradigms.distillation.loss import (
    distillation_loss_from_logits,
    make_distillation_loss,
)
from paradigms.models.cnn_teacher import TeacherCNN
from paradigms.models.mlp_student import StudentMLP
from paradigms.seed import set_seed
from paradigms.training.losses import hard_label_loss
from paradigms.training.trainer import fit
from tests.helpers import make_synthetic


def test_alpha_one_equals_hard_label_loss() -> None:
    """Verify alpha=1 is numerically identical to hard_label_loss across temperatures."""
    torch.manual_seed(42)
    s_logits = torch.randn(10, 10)
    labels = torch.randint(0, 10, (10,))

    for T in [0.5, 1.0, 2.0, 5.0]:
        dist_loss = distillation_loss_from_logits(
            s_logits=s_logits,
            t_logits=None,
            labels=labels,
            T=T,
            alpha=1.0,
        )
        hard_loss = hard_label_loss(s_logits, labels)
        assert torch.allclose(dist_loss, hard_loss, atol=1e-7)


def test_alpha_zero_identical_logits_zero_loss() -> None:
    """Verify alpha=0 yields loss < 1e-6 when student logits match teacher logits."""
    torch.manual_seed(42)
    logits = torch.randn(10, 10)
    loss = distillation_loss_from_logits(
        s_logits=logits,
        t_logits=logits,
        labels=None,
        T=2.0,
        alpha=0.0,
    )
    assert loss.item() < 1e-6


def test_manual_kl_formula_check() -> None:
    """Verify alpha=0, T=3 equals (p_t * (log p_t - log p_s)).sum(1).mean() * 9 within 1e-5."""
    torch.manual_seed(42)
    s_logits = torch.randn(16, 10)
    t_logits = torch.randn(16, 10)
    T = 3.0

    loss = distillation_loss_from_logits(
        s_logits=s_logits,
        t_logits=t_logits,
        labels=None,
        T=T,
        alpha=0.0,
    )

    p_t = F.softmax(t_logits / T, dim=-1)
    log_p_t = F.log_softmax(t_logits / T, dim=-1)
    log_p_s = F.log_softmax(s_logits / T, dim=-1)
    manual = (p_t * (log_p_t - log_p_s)).sum(1).mean() * (T * T)

    assert abs(loss.item() - manual.item()) < 1e-5


def test_alpha_zero_with_negative_one_labels() -> None:
    """Verify alpha=0 functions properly when labels = -1 (CE never computed)."""
    torch.manual_seed(42)
    s_logits = torch.randn(10, 10)
    t_logits = torch.randn(10, 10)
    labels = torch.full((10,), -1, dtype=torch.long)

    loss = distillation_loss_from_logits(
        s_logits=s_logits,
        t_logits=t_logits,
        labels=labels,
        T=2.0,
        alpha=0.0,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_real_teacher_student_gradient_flow() -> None:
    """Verify gradient flow and frozen teacher behavior with real TeacherCNN and StudentMLP."""
    teacher = TeacherCNN()
    teacher.train()  # Intentionally place in train mode before factory call

    student = StudentMLP(hidden=32)
    loss_fn = make_distillation_loss(teacher, T=2.0, alpha=0.5, device="cpu")

    # Teacher must be placed in eval mode
    assert teacher.training is False

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    s_logits = student(x)
    loss = loss_fn(s_logits, y, x)

    assert torch.isfinite(loss)
    loss.backward()

    # Student parameters must receive gradients
    for p in student.parameters():
        if p.requires_grad:
            assert p.grad is not None

    # Teacher parameters must have no gradients and remain frozen
    for p in teacher.parameters():
        assert p.grad is None
        assert not p.requires_grad

    assert teacher.training is False


def test_invalid_parameters_raise_value_error() -> None:
    """Verify invalid temperature (<= 0) or alpha (< 0 or > 1) raises ValueError."""
    teacher = TeacherCNN()

    # Invalid T
    with pytest.raises(ValueError, match="Temperature T must be positive"):
        distillation_loss_from_logits(torch.randn(2, 10), torch.randn(2, 10), torch.zeros(2, dtype=torch.long), T=0, alpha=0.5)

    with pytest.raises(ValueError, match="Temperature T must be positive"):
        make_distillation_loss(teacher, T=-1.0, alpha=0.5)

    # Invalid alpha
    with pytest.raises(ValueError, match="Alpha must be in"):
        distillation_loss_from_logits(torch.randn(2, 10), torch.randn(2, 10), torch.zeros(2, dtype=torch.long), T=1.0, alpha=-0.1)

    with pytest.raises(ValueError, match="Alpha must be in"):
        make_distillation_loss(teacher, T=1.0, alpha=1.1)


def test_fit_equivalence_on_synthetic_data() -> None:
    """Verify fit() with alpha=1 distillation loss matches hard_label_loss exactly."""
    teacher = TeacherCNN()
    cfg = {"epochs": 2, "lr": 0.01, "optimizer": "adam", "weight_decay": 0.0}

    # Run 1: Distillation with alpha=1.0
    set_seed(0)
    student1 = StudentMLP(32)
    ds1 = make_synthetic(100, seed=0)
    dev1 = make_synthetic(50, seed=0)
    loader1 = make_loader(ds1, batch_size=16, shuffle=True, seed=0)
    dev_loader1 = make_loader(dev1, batch_size=16, shuffle=False, seed=0)
    loss_fn1 = make_distillation_loss(teacher, T=2.0, alpha=1.0)
    res1 = fit(student1, loader1, dev_loader1, loss_fn1, cfg)

    # Run 2: Direct hard_label_loss
    set_seed(0)
    student2 = StudentMLP(32)
    ds2 = make_synthetic(100, seed=0)
    dev2 = make_synthetic(50, seed=0)
    loader2 = make_loader(ds2, batch_size=16, shuffle=True, seed=0)
    dev_loader2 = make_loader(dev2, batch_size=16, shuffle=False, seed=0)
    res2 = fit(student2, loader2, dev_loader2, hard_label_loss, cfg)

    losses1 = [h["train_loss"] for h in res1["history"]]
    losses2 = [h["train_loss"] for h in res2["history"]]
    assert losses1 == losses2
