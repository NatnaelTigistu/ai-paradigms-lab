"""Candidate verification mechanisms for recursive self-training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from paradigms.data.fashion_mnist import MEAN, STD
from paradigms.recursive.state import LoopBundle


class Verifier(ABC):
    """Abstract base class for candidate verification strategies in the recursive loop."""

    name: str

    def fit(
        self,
        labeled_images: Any,
        labeled_y: Any,
        pool_indices: Sequence[int] | np.ndarray,
        pool_images: Any,
    ) -> None:
        """Fit verifier state on labeled images and pre-compute pool predictions if applicable."""
        pass

    @abstractmethod
    def verify(
        self,
        indices: Sequence[int] | np.ndarray,
        preds: Sequence[int] | np.ndarray,
        confs: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return boolean mask indicating which predictions are verified."""
        raise NotImplementedError


class OracleVerifier(Verifier):
    """Simulates a checkable domain (such as formal verification or unit test suites) using hidden labels.

    Limitation: Real-world unlabeled pools lack access to ground-truth labels;
    this verifier serves as an idealized upper-bound benchmark rather than a viable production method.
    """

    def __init__(self, hidden_labels: Sequence[int] | np.ndarray) -> None:
        self.hidden_labels = np.asarray(hidden_labels, dtype=np.int64)
        self.name = "oracle"

    def verify(
        self,
        indices: Sequence[int] | np.ndarray,
        preds: Sequence[int] | np.ndarray,
        confs: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        indices_arr = np.asarray(indices, dtype=np.int64)
        preds_arr = np.asarray(preds, dtype=np.int64)

        if len(indices_arr) == 0:
            return np.zeros(0, dtype=bool)

        return preds_arr == self.hidden_labels[indices_arr]


class AgreementVerifier(Verifier):
    """Verifies candidates using an independent Random Forest trained on labeled data."""

    def __init__(self, threshold: float, seed: int) -> None:
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.name = "agreement"
        self._cached_rf_preds: dict[int, int] = {}

    def fit(
        self,
        labeled_images: Any,
        labeled_y: Any,
        pool_indices: Sequence[int] | np.ndarray,
        pool_images: Any,
    ) -> None:
        if isinstance(labeled_images, torch.Tensor):
            X_labeled = (
                labeled_images.detach()
                .cpu()
                .float()
                .reshape(len(labeled_images), -1)
                .numpy()
            )
        else:
            X_labeled = np.asarray(labeled_images, dtype=np.float32).reshape(
                len(labeled_images), -1
            )

        if isinstance(labeled_y, torch.Tensor):
            y_labeled = labeled_y.detach().cpu().numpy()
        else:
            y_labeled = np.asarray(labeled_y)

        if isinstance(pool_images, torch.Tensor):
            X_pool = (
                pool_images.detach()
                .cpu()
                .float()
                .reshape(len(pool_images), -1)
                .numpy()
            )
        else:
            X_pool = np.asarray(pool_images, dtype=np.float32).reshape(
                len(pool_images), -1
            )

        pool_idx_arr = np.asarray(pool_indices, dtype=np.int64)

        # an independent model family; NEVER retrained on pseudo-labels.
        rf = RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=-1)
        rf.fit(X_labeled, y_labeled)

        if len(pool_idx_arr) > 0 and len(X_pool) > 0:
            pool_preds = rf.predict(X_pool)
            self._cached_rf_preds = {
                int(idx): int(pred) for idx, pred in zip(pool_idx_arr, pool_preds)
            }
        else:
            self._cached_rf_preds = {}

    def verify(
        self,
        indices: Sequence[int] | np.ndarray,
        preds: Sequence[int] | np.ndarray,
        confs: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        indices_arr = np.asarray(indices, dtype=np.int64)
        preds_arr = np.asarray(preds, dtype=np.int64)
        confs_arr = np.asarray(confs, dtype=np.float64)

        if len(indices_arr) == 0:
            return np.zeros(0, dtype=bool)

        rf_labels = np.array(
            [self._cached_rf_preds.get(int(idx), -1) for idx in indices_arr],
            dtype=np.int64,
        )
        return (rf_labels == preds_arr) & (confs_arr >= self.threshold)


class SelfConfidenceVerifier(Verifier):
    """The generator grading itself via its predicted confidence (control)."""

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.name = "self_confidence"

    def verify(
        self,
        indices: Sequence[int] | np.ndarray,
        preds: Sequence[int] | np.ndarray,
        confs: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        confs_arr = np.asarray(confs, dtype=np.float64)
        if len(confs_arr) == 0:
            return np.zeros(0, dtype=bool)
        return confs_arr >= self.threshold


class NoVerifier(Verifier):
    """Null verifier that rejects all candidates (compute control)."""

    def __init__(self) -> None:
        self.name = "none"

    def verify(
        self,
        indices: Sequence[int] | np.ndarray,
        preds: Sequence[int] | np.ndarray,
        confs: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return np.zeros(len(indices), dtype=bool)


def build_verifier(
    name: str,
    rcfg: dict[str, Any] | Any,
    bundle: LoopBundle,
    seed: int,
) -> Verifier:
    """Build and fit a verifier instance for the recursive loop.

    Structural Enforcement:
        Only OracleVerifier has access to bundle.hidden_labels. All other verifiers
        receive zero access to hidden pool labels.
    """
    key = name.lower()
    cfg = rcfg.get("recursive", rcfg) if hasattr(rcfg, "get") else rcfg
    agreement_threshold = (
        cfg.get("agreement_threshold", 0.9)
        if hasattr(cfg, "get")
        else getattr(cfg, "agreement_threshold", 0.9)
    )
    self_conf_threshold = (
        cfg.get("self_conf_threshold", 0.9)
        if hasattr(cfg, "get")
        else getattr(cfg, "self_conf_threshold", 0.9)
    )

    if key == "oracle":
        return OracleVerifier(hidden_labels=bundle.hidden_labels)

    if key == "agreement":
        verifier = AgreementVerifier(threshold=agreement_threshold, seed=seed)
        # Extract images and labels without reading bundle.hidden_labels
        labeled_indices = bundle.partition.labeled
        pool_indices = bundle.partition.pool

        if hasattr(bundle.base, "data"):
            raw_labeled = bundle.base.data[labeled_indices]
            raw_pool = bundle.base.data[pool_indices]
            labeled_images = (
                ((raw_labeled.float() / 255.0 - MEAN) / STD)
                .reshape(len(labeled_indices), -1)
                .numpy()
            )
            pool_images = (
                ((raw_pool.float() / 255.0 - MEAN) / STD)
                .reshape(len(pool_indices), -1)
                .numpy()
            )
        else:
            labeled_images = np.stack(
                [bundle.base[i][0].reshape(-1).numpy() for i in labeled_indices]
            )
            pool_images = np.stack(
                [bundle.base[i][0].reshape(-1).numpy() for i in pool_indices]
            )

        if hasattr(bundle.base, "targets"):
            targets = bundle.base.targets
            if isinstance(targets, torch.Tensor):
                labeled_y = targets[labeled_indices].cpu().numpy()
            else:
                labeled_y = np.asarray(targets)[labeled_indices]
        else:
            labeled_y = np.array([bundle.base[i][1] for i in labeled_indices])

        verifier.fit(
            labeled_images=labeled_images,
            labeled_y=labeled_y,
            pool_indices=pool_indices,
            pool_images=pool_images,
        )
        return verifier

    if key in ("self_confidence", "self_conf"):
        return SelfConfidenceVerifier(threshold=self_conf_threshold)

    if key in ("none", "no_verifier", "null"):
        return NoVerifier()

    raise ValueError(
        f"Unknown verifier '{name}'. Valid choices are: 'oracle', 'agreement', 'self_confidence', 'none'."
    )
