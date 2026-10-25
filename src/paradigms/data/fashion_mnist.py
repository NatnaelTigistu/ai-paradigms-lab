"""Fashion-MNIST dataset management, test-set leakage guard, and data loaders.

Fashion-MNIST comprises 60,000 training images and 10,000 official test images.
We split the 60,000 training samples into 50,000 "train" (for model parameter
optimization) and 10,000 "dev" (for validation and hyperparameter tuning).
The official 10,000 test images represent the protected final evaluation benchmark
and may only be accessed when explicitly permitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from paradigms.data.splits import (
    load_splits,
    make_splits,
    save_splits,
    stratified_subset,
)
from paradigms.seed import make_generator

MEAN: float = 0.2860
STD: float = 0.3530


def _transform() -> transforms.Compose:
    """Standard pre-processing pipeline for Fashion-MNIST images.

    Converts grayscale PIL images in [0, 255] to torch.FloatTensor in [0.0, 1.0],
    then normalizes with dataset-wide training mean and standard deviation.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,)),
        ]
    )


class FashionData:
    """Manages Fashion-MNIST datasets with guarded split access.

    Guards against inadvertent test set leakage by raising a PermissionError
    if the test set is requested without explicit authorization.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        split_seed: int = 42,
        dev_size: int = 10000,
    ) -> None:
        self.root = Path(root)
        self.split_seed = split_seed
        self.dev_size = dev_size

        transform = _transform()
        self.full_train = torchvision.datasets.FashionMNIST(
            root=str(self.root),
            train=True,
            download=True,
            transform=transform,
        )
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root=str(self.root),
            train=False,
            download=True,
            transform=transform,
        )

        self.labels = np.asarray(self.full_train.targets)

        split_path = self.root / "splits" / f"fashion_split_{split_seed}_{dev_size}.json"
        if split_path.exists():
            self.splits = load_splits(split_path)
        else:
            self.splits = make_splits(
                n_total=len(self.full_train),
                dev_size=dev_size,
                split_seed=split_seed,
            )
            save_splits(self.splits, split_path)

    def get(self, name: str, allow_test: bool = False) -> Dataset:
        """Retrieve a specific partition of the dataset.

        Args:
            name: Partition name ("train", "dev", or "test").
            allow_test: Explicit confirmation required to access the test set.

        Raises:
            PermissionError: If `name` is "test" and `allow_test` is False.
            ValueError: If `name` is unrecognized.

        Returns:
            PyTorch Dataset / Subset corresponding to the requested partition.
        """
        if name in ("train", "dev"):
            return Subset(self.full_train, self.splits[name])
        elif name == "test":
            if not allow_test:
                raise PermissionError(
                    "Test set is locked. Pass allow_test=True only for the final evaluation."
                )
            return self.test_dataset
        else:
            raise ValueError(f"Unknown split name '{name}'. Expected 'train', 'dev', or 'test'.")

    def train_indices_subset(self, fraction: float, seed: int) -> Subset:
        """Generate a stratified, nested subset of the training set.

        Args:
            fraction: Fraction of training data to include in (0.0, 1.0].
            seed: Explicit seed for deterministic class-wise ordering.

        Returns:
            torch.utils.data.Subset wrapping the chosen training indices.
        """
        subset_indices = stratified_subset(
            indices=self.splits["train"],
            labels=self.labels,
            fraction=fraction,
            seed=seed,
        )
        return Subset(self.full_train, subset_indices)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a reproducible PyTorch DataLoader for CPU training.

    Explicitly passes a seeded torch.Generator to govern batch shuffling,
    avoiding reliance on global torch RNG state. num_workers is set to 0 for
    simple, deterministic CPU execution.
    """
    generator = make_generator(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )
