"""Tests for recursive loop execution, acceptance gating, and rollback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset

import paradigms.recursive.loop as loop_module
from paradigms.recursive import (
    LoopBundle,
    LoopPartition,
    decide_gate,
    run_loop,
)
from paradigms.recursive.datasets import PseudoLabeledDataset


def test_decide_gate_cases() -> None:
    """Verify decide_gate logic: improvement, small gain, class drop, None values."""
    baseline = {"accuracy": 0.80, "per_class_accuracy": [0.80, 0.80]}
    gate_cfg = {"min_improvement": 0.01, "max_class_drop": 0.05}

    # 1. Improvement accepted
    candidate_good = {"accuracy": 0.83, "per_class_accuracy": [0.82, 0.84]}
    ok, reason = decide_gate(candidate_good, baseline, gate_cfg)
    assert ok is True
    assert reason == "accepted"

    # 2. Small gain rejected
    candidate_small = {"accuracy": 0.805, "per_class_accuracy": [0.805, 0.805]}
    ok, reason = decide_gate(candidate_small, baseline, gate_cfg)
    assert ok is False
    assert "below 0.01" in reason

    # 3. Class drop rejected even with higher overall accuracy
    candidate_drop = {"accuracy": 0.85, "per_class_accuracy": [0.99, 0.71]}  # Class 1 dropped by 0.09
    ok, reason = decide_gate(candidate_drop, baseline, gate_cfg)
    assert ok is False
    assert "class 1 dropped 0.090" in reason

    # 4. None entries skipped
    baseline_none = {"accuracy": 0.80, "per_class_accuracy": [None, 0.80]}
    candidate_skip_none = {"accuracy": 0.83, "per_class_accuracy": [0.50, 0.83]}
    ok, reason = decide_gate(candidate_skip_none, baseline_none, gate_cfg)
    assert ok is True
    assert reason == "accepted"


class SyntheticDataset:
    """In-memory synthetic dataset providing .data and .targets."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = ((self.data[idx].float() / 255.0 - 0.286) / 0.353).unsqueeze(0)
        return x, int(self.targets[idx])


def make_synthetic_bundle(
    n_labeled: int = 40,
    pool_size: int = 60,
    gate_size: int = 20,
    monitor_size: int = 20,
    seed: int = 42,
) -> LoopBundle:
    """Create a self-contained synthetic LoopBundle for quick unit testing."""
    n_total = n_labeled + pool_size + gate_size + monitor_size
    data_tensor = torch.randint(0, 256, (n_total, 28, 28), dtype=torch.uint8)
    targets_tensor = torch.tensor([i % 2 for i in range(n_total)], dtype=torch.long)

    base = SyntheticDataset(data_tensor, targets_tensor)
    indices = list(range(n_total))

    labeled = indices[:n_labeled]
    pool = indices[n_labeled : n_labeled + pool_size]
    gate = indices[n_labeled + pool_size : n_labeled + pool_size + gate_size]
    monitor = indices[n_labeled + pool_size + gate_size :]

    partition = LoopPartition(
        labeled=labeled,
        pool=pool,
        gate=gate,
        monitor=monitor,
    )

    return LoopBundle(
        base=base,
        hidden_labels=targets_tensor.numpy().copy(),
        partition=partition,
        gate_ds=Subset(base, gate),
        monitor_ds=Subset(base, monitor),
    )


def test_oracle_loop_precision(tmp_path: Path) -> None:
    """Verify recursive loop with oracle verifier always logs precision == 1.0."""
    bundle = make_synthetic_bundle(seed=1)
    cfg = {
        "recursive": {
            "model": "small_cnn",
            "model_kwargs": {},
            "rounds": 2,
            "epochs_initial": 1,
            "epochs_per_round": 1,
            "lr_round": 0.01,
            "batch_size": 16,
            "cap_per_class": 5,
            "curation": "random",
            "agreement_threshold": 0.5,
            "self_conf_threshold": 0.5,
            "gate": {"enabled": False, "patience": 2},
        },
        "training": {"lr": 0.01, "optimizer": "adam", "weight_decay": 0.0},
    }

    res = run_loop(
        cfg,
        bundle,
        seed=42,
        verifier_name="oracle",
        gate_enabled=False,
        ckpt_dir=tmp_path / "ckpts",
    )

    log = res["log"]
    assert len(log) >= 2  # Round 0 and at least Round 1
    for row in log[1:]:
        if row["n_selected"] > 0:
            assert row["selected_precision"] == 1.0


def test_rollback_on_impossible_gate(tmp_path: Path) -> None:
    """Verify rollback discards candidates and restores exact round0 weights on rejection."""
    bundle = make_synthetic_bundle(seed=2)
    cfg = {
        "recursive": {
            "model": "small_cnn",
            "model_kwargs": {},
            "rounds": 3,
            "epochs_initial": 1,
            "epochs_per_round": 1,
            "lr_round": 0.01,
            "batch_size": 16,
            "cap_per_class": 5,
            "curation": "random",
            "gate": {
                "enabled": True,
                "min_improvement": 10.0,  # Impossible to achieve
                "max_class_drop": 0.01,
                "patience": 2,
            },
        },
        "training": {"lr": 0.01, "optimizer": "adam", "weight_decay": 0.0},
    }

    ckpt_dir = tmp_path / "rollback_ckpts"
    res = run_loop(
        cfg,
        bundle,
        seed=10,
        verifier_name="oracle",
        gate_enabled=True,
        ckpt_dir=ckpt_dir,
    )

    assert res["stop_reason"] == "patience"
    assert res["n_pseudo_labels_total"] == 0
    assert res["n_rejected_rounds"] == 2

    # Verify final checkpoint matches round0 checkpoint exactly
    round0_state = torch.load(ckpt_dir / "round0.pt", weights_only=True)
    final_state = torch.load(ckpt_dir / "final.pt", weights_only=True)

    for k in round0_state:
        assert torch.equal(round0_state[k], final_state[k])


def test_accept_everything_verifier_captures_training_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify pseudo-labels match model predictions and logged precision < 1."""
    bundle = make_synthetic_bundle(seed=3)
    cfg = {
        "recursive": {
            "model": "small_cnn",
            "model_kwargs": {},
            "rounds": 1,
            "epochs_initial": 1,
            "epochs_per_round": 1,
            "lr_round": 0.01,
            "batch_size": 16,
            "cap_per_class": 10,
            "curation": "random",
            "self_conf_threshold": 0.0,  # Accepts everything
            "gate": {"enabled": False, "patience": 2},
        },
        "training": {"lr": 0.01, "optimizer": "adam", "weight_decay": 0.0},
    }

    captured_pseudo_datasets: list[PseudoLabeledDataset] = []
    models_at_fit: list[Any] = []
    original_fit = loop_module.fit
    original_make_fast_loader = loop_module.make_fast_loader

    def mock_make_fast_loader(dataset, batch_size=128, shuffle=False, seed=0):
        if isinstance(dataset, ConcatDataset):
            for part in dataset.datasets:
                if isinstance(part, PseudoLabeledDataset):
                    captured_pseudo_datasets.append(part)
        return original_make_fast_loader(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)

    def mock_fit(model, train_loader, dev_loader=None, loss_fn=None, training_cfg=None, device="cpu", verbose=False):
        models_at_fit.append(copy.deepcopy(model))
        return original_fit(model, train_loader, dev_loader, loss_fn, training_cfg, device, verbose)

    import copy
    monkeypatch.setattr(loop_module, "make_fast_loader", mock_make_fast_loader)
    monkeypatch.setattr(loop_module, "fit", mock_fit)

    res = run_loop(
        cfg,
        bundle,
        seed=5,
        verifier_name="self_confidence",
        gate_enabled=False,
    )

    log = res["log"]
    assert len(log) == 2  # Round 0 + Round 1
    # On random synthetic data with threshold=0, precision should be < 1.0
    assert log[1]["selected_precision"] < 1.0

    # Verify captured pseudo-labels match the model's predictions before round 1 training
    assert len(captured_pseudo_datasets) == 1
    pseudo_ds = captured_pseudo_datasets[0]
    assert len(pseudo_ds) > 0

    # Model before round 1 fit is models_at_fit[1] (first is round 0 model)
    model_before_round1 = models_at_fit[1]
    model_before_round1.eval()

    with torch.no_grad():
        for idx, assigned_label in zip(pseudo_ds.indices, pseudo_ds.labels):
            x, _ = bundle.base[idx]
            pred = model_before_round1(x.unsqueeze(0)).argmax(dim=-1).item()
            assert assigned_label == pred, f"Assigned label {assigned_label} != predicted {pred}"


def test_determinism_same_seed() -> None:
    """Verify running the loop with the same seed produces identical log records."""
    bundle = make_synthetic_bundle(seed=4)
    cfg = {
        "recursive": {
            "model": "small_cnn",
            "model_kwargs": {},
            "rounds": 2,
            "epochs_initial": 1,
            "epochs_per_round": 1,
            "lr_round": 0.01,
            "batch_size": 16,
            "cap_per_class": 5,
            "curation": "random",
            "gate": {"enabled": False, "patience": 2},
        },
        "training": {"lr": 0.01, "optimizer": "adam", "weight_decay": 0.0},
    }

    res1 = run_loop(cfg, bundle, seed=77, verifier_name="self_confidence", gate_enabled=False)
    res2 = run_loop(cfg, bundle, seed=77, verifier_name="self_confidence", gate_enabled=False)

    keys_to_compare = [
        "round",
        "accepted",
        "reason",
        "n_candidates",
        "n_verified",
        "n_selected",
        "selected_precision",
        "train_size",
        "cand_gate_acc",
        "cand_monitor_acc",
        "current_gate_acc",
        "current_monitor_acc",
        "label_noise_in_training",
    ]

    assert len(res1["log"]) == len(res2["log"])
    for r1, r2 in zip(res1["log"], res2["log"]):
        for k in keys_to_compare:
            assert r1[k] == r2[k], f"Mismatch on key '{k}': {r1[k]} vs {r2[k]}"


def test_gate_and_monitor_never_leak_into_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify gate and monitor sample indices never appear in any training loader dataset."""
    bundle = make_synthetic_bundle(seed=5)
    gate_set = set(bundle.partition.gate)
    monitor_set = set(bundle.partition.monitor)

    cfg = {
        "recursive": {
            "model": "small_cnn",
            "model_kwargs": {},
            "rounds": 2,
            "epochs_initial": 1,
            "epochs_per_round": 1,
            "lr_round": 0.01,
            "batch_size": 16,
            "cap_per_class": 5,
            "curation": "random",
            "gate": {"enabled": True, "min_improvement": 0.0001, "max_class_drop": 0.1, "patience": 2},
        },
        "training": {"lr": 0.01, "optimizer": "adam", "weight_decay": 0.0},
    }

    original_make_fast_loader = loop_module.make_fast_loader

    def tracking_make_fast_loader(dataset, batch_size=128, shuffle=False, seed=0):
        # Only verify training loaders (which have shuffle=True)
        if shuffle:
            if isinstance(dataset, Subset):
                ds_indices = set(dataset.indices)
                assert ds_indices.isdisjoint(gate_set), "Gate indices leaked into training Subset!"
                assert ds_indices.isdisjoint(monitor_set), "Monitor indices leaked into training Subset!"
            elif isinstance(dataset, ConcatDataset):
                for part in dataset.datasets:
                    if isinstance(part, Subset):
                        ds_indices = set(part.indices)
                        assert ds_indices.isdisjoint(gate_set), "Gate indices leaked into ConcatDataset!"
                        assert ds_indices.isdisjoint(monitor_set), "Monitor indices leaked into ConcatDataset!"
                    elif isinstance(part, PseudoLabeledDataset):
                        ds_indices = set(part.indices)
                        assert ds_indices.isdisjoint(gate_set), "Gate indices leaked into PseudoLabeledDataset!"
                        assert ds_indices.isdisjoint(monitor_set), "Monitor indices leaked into PseudoLabeledDataset!"
        return original_make_fast_loader(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)

    monkeypatch.setattr(loop_module, "make_fast_loader", tracking_make_fast_loader)

    run_loop(cfg, bundle, seed=88, verifier_name="oracle", gate_enabled=True)
