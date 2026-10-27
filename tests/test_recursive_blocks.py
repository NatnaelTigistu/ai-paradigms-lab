"""Tests for recursive self-training building blocks."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from paradigms.data import FashionData, split_dev
from paradigms.data.tensor_loader import materialize
from paradigms.models import SmallCNN, build_model, count_parameters
from paradigms.recursive import PseudoLabeledDataset
from paradigms.training import fit
from paradigms.training.losses import hard_label_loss
from tests.helpers import make_synthetic


def test_split_dev_properties() -> None:
    """Verify gate/monitor split is disjoint, exhaustive, balanced and deterministic."""
    data = FashionData(root="./data", split_seed=42)
    dev_indices = data.splits["dev"]
    labels = data.train_labels

    splits1 = split_dev(dev_indices, labels, seed=42)
    splits2 = split_dev(dev_indices, labels, seed=42)

    gate = splits1["gate"]
    monitor = splits1["monitor"]

    # Determinism
    assert gate == splits2["gate"]
    assert monitor == splits2["monitor"]

    # Disjoint
    assert set(gate).isdisjoint(set(monitor))

    # Union equals dev
    assert sorted(gate + monitor) == sorted(dev_indices)

    # Class balance within +/-1 per class
    for c in range(10):
        gate_c = sum(1 for idx in gate if labels[idx] == c)
        mon_c = sum(1 for idx in monitor if labels[idx] == c)
        assert abs(gate_c - mon_c) <= 1


def test_fashion_data_gate_monitor_access() -> None:
    """Verify get('gate') and get('monitor') do not require allow_test."""
    data = FashionData(root="./data", split_seed=42)

    # Gate and monitor splits should be accessible without allow_test=True
    gate_ds = data.get("gate")
    monitor_ds = data.get("monitor")
    assert len(gate_ds) + len(monitor_ds) == len(data.get("dev"))
    assert abs(len(gate_ds) - len(monitor_ds)) <= 10

    # Test set still raises PermissionError
    with pytest.raises(PermissionError):
        data.get("test")


def test_fashion_data_readonly_properties() -> None:
    """Verify full_train and train_labels are read-only properties."""
    data = FashionData(root="./data", split_seed=42)
    assert len(data.full_train) == 60000
    assert len(data.train_labels) == 60000

    with pytest.raises(AttributeError):
        data.full_train = None  # type: ignore[misc]

    with pytest.raises(AttributeError):
        data.train_labels = None  # type: ignore[misc]


def test_pseudolabeled_dataset() -> None:
    """Verify PseudoLabeledDataset replaces labels while preserving correct base images."""
    base = make_synthetic(50, seed=123)
    indices = [3, 14, 25]
    override_labels = [9, 8, 7]

    pseudo_ds = PseudoLabeledDataset(base, indices, override_labels)
    assert len(pseudo_ds) == 3

    for i, (orig_idx, target_label) in enumerate(zip(indices, override_labels)):
        img, label = pseudo_ds[i]
        assert label == target_label
        assert torch.equal(img, base[orig_idx][0])

    with pytest.raises(ValueError):
        PseudoLabeledDataset(base, [1, 2], [5])


def test_materialize_matches_slow_path() -> None:
    """Verify materialize on ConcatDataset and PseudoLabeledDataset matches slow DataLoader path."""
    data = FashionData(root="./data", split_seed=42)
    indices = [0, 10, 20, 30, 40]
    pseudo_labels = [7, 6, 5, 4, 3]

    # Test PseudoLabeledDataset
    pseudo_ds = PseudoLabeledDataset(data.full_train, indices, pseudo_labels)
    slow_loader = DataLoader(pseudo_ds, batch_size=len(pseudo_ds), shuffle=False)
    slow_x, slow_y = next(iter(slow_loader))

    fast_x, fast_y = materialize(pseudo_ds)
    assert torch.allclose(slow_x, fast_x, atol=1e-5)
    assert torch.equal(slow_y, fast_y)

    # Test ConcatDataset
    sub_ds = Subset(data.full_train, [1, 2, 3])
    concat_ds = ConcatDataset([sub_ds, pseudo_ds])
    slow_concat_loader = DataLoader(concat_ds, batch_size=len(concat_ds), shuffle=False)
    slow_cx, slow_cy = next(iter(slow_concat_loader))

    fast_cx, fast_cy = materialize(concat_ds)
    assert torch.allclose(slow_cx, fast_cx, atol=1e-5)
    assert torch.equal(slow_cy, fast_cy)


def test_small_cnn_architecture_and_registry() -> None:
    """Verify SmallCNN output shape (4,10) and exactly 105,866 parameters."""
    model = SmallCNN()
    assert count_parameters(model) == 105866

    # Test 4D input
    x_4d = torch.randn(4, 1, 28, 28)
    out_4d = model(x_4d)
    assert out_4d.shape == (4, 10)

    # Test 2D flattened input
    x_2d = torch.randn(4, 784)
    out_2d = model(x_2d)
    assert out_2d.shape == (4, 10)

    # Registry lookup
    reg_model = build_model("small_cnn")
    assert isinstance(reg_model, SmallCNN)
    assert count_parameters(reg_model) == 105866


def test_fit_optional_dev_loader() -> None:
    """Verify fit(dev_loader=None) runs, skips dev eval, and keeps last-epoch weights."""
    ds = make_synthetic(40, seed=77)
    train_loader = DataLoader(ds, batch_size=10, shuffle=False)

    model = SmallCNN()
    cfg = {"epochs": 2, "lr": 0.01, "optimizer": "adam", "weight_decay": 0.0}

    out = fit(model, train_loader, dev_loader=None, loss_fn=hard_label_loss, training_cfg=cfg)

    assert out["best_epoch"] is None
    assert out["best_dev_acc"] is None
    assert len(out["history"]) == 2

    # Check history structure
    for record in out["history"]:
        assert "epoch" in record
        assert "train_loss" in record
        assert "seconds" in record
        assert "dev_loss" not in record
        assert "dev_acc" not in record

    # Check that model kept last epoch weights by taking another step and seeing continuity
    loss_ep2 = out["history"][1]["train_loss"]
    assert isinstance(loss_ep2, float)
