"""Tests for Fashion-MNIST loading, fixed splits, test set guard, and stratified subsets."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from paradigms.data.fashion_mnist import FashionData, make_loader
from paradigms.data.splits import (
    load_splits,
    make_splits,
    save_splits,
    stratified_subset,
)


@pytest.fixture(scope="module")
def fashion_data() -> FashionData:
    """Fixture providing FashionData initialized once per test module.

    Uses root="./data" so that downloaded Fashion-MNIST binary files are cached
    locally and shared across test runs.
    """
    return FashionData(root="./data", split_seed=42, dev_size=10000)


def test_split_lengths(fashion_data: FashionData) -> None:
    """Verify that the 60,000 raw training pool is split into 50k train and 10k dev."""
    train_ds = fashion_data.get("train")
    dev_ds = fashion_data.get("dev")
    assert len(train_ds) == 50000
    assert len(dev_ds) == 10000


def test_disjoint_and_exhaustive_splits(fashion_data: FashionData) -> None:
    """Verify train and dev index sets are disjoint and fully cover range(60,000)."""
    train_idx = set(fashion_data.splits["train"])
    dev_idx = set(fashion_data.splits["dev"])

    assert train_idx.isdisjoint(dev_idx)
    assert train_idx | dev_idx == set(range(60000))


def test_split_seed_reproducibility() -> None:
    """Verify same split_seed produces identical splits, while different seeds differ."""
    s1 = make_splits(n_total=60000, dev_size=10000, split_seed=42)
    s2 = make_splits(n_total=60000, dev_size=10000, split_seed=42)
    s3 = make_splits(n_total=60000, dev_size=10000, split_seed=99)

    assert s1["train"] == s2["train"]
    assert s1["dev"] == s2["dev"]
    assert s1["train"] != s3["train"]
    assert s1["dev"] != s3["dev"]


def test_test_set_guard(fashion_data: FashionData) -> None:
    """Verify accessing the test set without allow_test=True raises PermissionError."""
    with pytest.raises(PermissionError, match="Test set is locked"):
        fashion_data.get("test")

    with pytest.raises(PermissionError):
        fashion_data.get("test", allow_test=False)

    test_ds = fashion_data.get("test", allow_test=True)
    assert len(test_ds) == 10000


def test_invalid_split_name(fashion_data: FashionData) -> None:
    """Verify unrecognized split names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown split name"):
        fashion_data.get("validation")


def test_sample_shapes_and_types(fashion_data: FashionData) -> None:
    """Verify image tensor shape is (1, 28, 28) and labels are ints in [0, 9]."""
    train_ds = fashion_data.get("train")
    img, label = train_ds[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 28, 28)
    assert isinstance(label, int)
    assert 0 <= label <= 9


def test_stratified_subset_class_balance(fashion_data: FashionData) -> None:
    """Verify stratified_subset(fraction=0.1) gives ~5000 train samples.

    Because make_splits creates an unstratified random split of 50,000 samples,
    the train split exhibits slight natural class fluctuations (4984 to 5053 per class),
    yielding ~5000 samples (5006 samples with per-class counts 499 to 506).
    On a balanced index pool, per-class counts are within +/-1 of each other.
    """
    sub = fashion_data.train_indices_subset(fraction=0.1, seed=42)
    assert abs(len(sub) - 5000) <= 10

    train_indices = fashion_data.splits["train"]
    labels = [int(fashion_data.labels[i]) for i in sub.indices]
    counts = [labels.count(c) for c in range(10)]

    for c in range(10):
        expected_c = math.ceil(0.1 * sum(1 for i in train_indices if int(fashion_data.labels[i]) == c))
        assert counts[c] == expected_c
    assert max(counts) - min(counts) <= 10

    # On a balanced index pool, verify per-class counts are within +/-1 of each other
    balanced_pool: list[int] = []
    for c in range(10):
        c_idxs = [i for i in train_indices if int(fashion_data.labels[i]) == c]
        balanced_pool.extend(c_idxs[:500])
    sub_balanced = stratified_subset(balanced_pool, fashion_data.labels, fraction=0.1, seed=42)
    b_labels = [int(fashion_data.labels[i]) for i in sub_balanced]
    b_counts = [b_labels.count(c) for c in range(10)]
    assert max(b_counts) - min(b_counts) <= 1


def test_stratified_subset_nesting(fashion_data: FashionData) -> None:
    """Verify smaller subsets are exact subsets of larger subsets with the same seed."""
    train_indices = fashion_data.splits["train"]
    sub_5 = set(
        stratified_subset(
            indices=train_indices,
            labels=fashion_data.labels,
            fraction=0.05,
            seed=3,
        )
    )
    sub_10 = set(
        stratified_subset(
            indices=train_indices,
            labels=fashion_data.labels,
            fraction=0.10,
            seed=3,
        )
    )

    assert sub_5.issubset(sub_10)
    assert len(sub_5) < len(sub_10)


def test_stratified_subset_fraction_one(fashion_data: FashionData) -> None:
    """Verify fraction=1.0 returns every index."""
    train_indices = fashion_data.splits["train"]
    full_sub = stratified_subset(
        indices=train_indices,
        labels=fashion_data.labels,
        fraction=1.0,
        seed=3,
    )

    assert full_sub == sorted(train_indices)


def test_save_and_load_splits(tmp_path: Path) -> None:
    """Verify split dictionary serialization and deserialization via JSON."""
    splits = make_splits(n_total=100, dev_size=20, split_seed=123)
    p = tmp_path / "splits.json"
    save_splits(splits, p)
    loaded = load_splits(p)
    assert loaded == splits


def test_make_loader(fashion_data: FashionData) -> None:
    """Verify DataLoader creation with explicit seed and deterministic batch generation."""
    train_ds = fashion_data.get("train")
    loader = make_loader(train_ds, batch_size=32, shuffle=True, seed=42)
    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (32, 1, 28, 28)
    assert batch_y.shape == (32,)
