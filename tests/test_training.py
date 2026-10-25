"""Tests for pluggable losses, evaluation routines, and generic trainer."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from paradigms.data.fashion_mnist import make_loader
from paradigms.models.mlp_student import StudentMLP
from paradigms.seed import set_seed
from paradigms.training.evaluate import evaluate, metrics_from_logits
from paradigms.training.losses import hard_label_loss
from paradigms.training.trainer import fit
from tests.helpers import make_synthetic


def test_hard_label_loss_equivalence() -> None:
    """Verify hard_label_loss matches PyTorch F.cross_entropy with and without inputs arg."""
    logits = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))
    inputs = torch.randn(8, 1, 28, 28)

    expected = F.cross_entropy(logits, labels)
    assert torch.equal(hard_label_loss(logits, labels, inputs), expected)
    assert torch.equal(hard_label_loss(logits, labels), expected)


def test_metrics_from_logits_handmade() -> None:
    """Verify metrics_from_logits with controlled predictions.

    Labels: [0, 0, 1, 1], Predictions: [0, 1, 1, 1]
    Accuracy: 3/4 = 0.75
    Class 0 acc: 1/2 = 0.5
    Class 1 acc: 2/2 = 1.0
    Class 2: None (0 samples)
    """
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    logits = torch.zeros(4, 10)
    logits[0, 0] = 5.0  # pred 0, label 0 (correct)
    logits[1, 1] = 5.0  # pred 1, label 0 (incorrect)
    logits[2, 1] = 5.0  # pred 1, label 1 (correct)
    logits[3, 1] = 5.0  # pred 1, label 1 (correct)

    metrics = metrics_from_logits(logits, labels, num_classes=10)
    assert metrics["accuracy"] == 0.75
    assert metrics["num_samples"] == 4
    assert metrics["per_class_accuracy"][0] == 0.5
    assert metrics["per_class_accuracy"][1] == 1.0
    assert metrics["per_class_accuracy"][2] is None


def test_evaluate_preserves_training_mode() -> None:
    """Verify evaluate() restores model.training state for both train and eval modes."""
    model = StudentMLP(32)
    ds = make_synthetic(20, seed=42)
    loader = DataLoader(ds, batch_size=10)

    # When starting in train mode
    model.train()
    assert model.training is True
    evaluate(model, loader)
    assert model.training is True

    # When starting in eval mode
    model.eval()
    assert model.training is False
    evaluate(model, loader)
    assert model.training is False


def test_fit_synthetic_accuracy() -> None:
    """Verify fit() trains StudentMLP(32) on synthetic data to best_dev_acc > 0.9 in 3 epochs."""
    train_ds = make_synthetic(300, seed=1)
    dev_ds = make_synthetic(100, seed=2)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False)

    model = StudentMLP(32)
    cfg = {"epochs": 3, "lr": 0.01, "optimizer": "adam", "weight_decay": 0.0}
    result = fit(model, train_loader, dev_loader, hard_label_loss, cfg)

    assert result["best_dev_acc"] > 0.9
    assert result["best_epoch"] in [1, 2, 3]
    assert len(result["history"]) == 3


def test_fit_determinism() -> None:
    """Verify identical seeds produce identical train_loss histories."""
    cfg = {"epochs": 2, "lr": 0.01, "optimizer": "adam", "weight_decay": 0.0}

    # Run 1
    set_seed(0)
    model1 = StudentMLP(32)
    ds1 = make_synthetic(100, seed=0)
    dev1 = make_synthetic(50, seed=0)
    loader1 = make_loader(ds1, batch_size=16, shuffle=True, seed=0)
    dev_loader1 = make_loader(dev1, batch_size=16, shuffle=False, seed=0)
    res1 = fit(model1, loader1, dev_loader1, hard_label_loss, cfg)

    # Run 2
    set_seed(0)
    model2 = StudentMLP(32)
    ds2 = make_synthetic(100, seed=0)
    dev2 = make_synthetic(50, seed=0)
    loader2 = make_loader(ds2, batch_size=16, shuffle=True, seed=0)
    dev_loader2 = make_loader(dev2, batch_size=16, shuffle=False, seed=0)
    res2 = fit(model2, loader2, dev_loader2, hard_label_loss, cfg)

    losses1 = [h["train_loss"] for h in res1["history"]]
    losses2 = [h["train_loss"] for h in res2["history"]]
    assert losses1 == losses2


def test_best_state_restoration() -> None:
    """Verify that model weights after fit() match the best development epoch's accuracy."""
    train_ds = make_synthetic(100, seed=10)
    dev_ds = make_synthetic(50, seed=20)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False)

    model = StudentMLP(32)
    cfg = {"epochs": 3, "lr": 0.01, "optimizer": "adam", "weight_decay": 0.0}
    res = fit(model, train_loader, dev_loader, hard_label_loss, cfg)

    current_dev_acc = evaluate(model, dev_loader)["accuracy"]
    max_history_acc = max(h["dev_acc"] for h in res["history"])

    assert current_dev_acc == max_history_acc
    assert current_dev_acc == res["best_dev_acc"]


def test_custom_loss_plumbing_spy() -> None:
    """Verify custom loss_fn receives raw inputs batch with shape (batch, 1, 28, 28)."""
    recorded_shapes: list[torch.Size] = []

    def spy_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        assert inputs is not None
        recorded_shapes.append(inputs.shape)
        return hard_label_loss(logits, labels)

    train_ds = make_synthetic(32, seed=1)
    dev_ds = make_synthetic(16, seed=2)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False)

    model = StudentMLP(32)
    cfg = {"epochs": 1, "lr": 0.01, "optimizer": "adam", "weight_decay": 0.0}
    fit(model, train_loader, dev_loader, spy_loss, cfg)

    assert len(recorded_shapes) > 0
    assert all(shape[1:] == torch.Size([1, 28, 28]) for shape in recorded_shapes)


def test_unknown_optimizer_raises() -> None:
    """Verify fit raises ValueError when an unknown optimizer name is provided."""
    train_ds = make_synthetic(16, seed=1)
    dev_ds = make_synthetic(16, seed=2)
    train_loader = DataLoader(train_ds, batch_size=16)
    dev_loader = DataLoader(dev_ds, batch_size=16)

    model = StudentMLP(32)
    cfg = {"epochs": 1, "lr": 0.01, "optimizer": "rmsprop"}
    with pytest.raises(ValueError, match="Unknown optimizer 'rmsprop'"):
        fit(model, train_loader, dev_loader, hard_label_loss, cfg)
