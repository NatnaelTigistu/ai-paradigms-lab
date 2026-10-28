#!/usr/bin/env python3
"""Verify dataset splits, assert non-leakage, and export docs/split_manifest.json.

Builds FashionData exactly like the training experiments, verifies disjointness across
train, dev, gate, monitor, and test partitions, computes deterministic sha256 hashes
for sorted index lists, and exports a verified manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradigms.config import load_config
from paradigms.data.fashion_mnist import FashionData
from paradigms.recursive.state import assert_no_leakage, make_partition


def compute_sha256(indices: Sequence[int]) -> str:
    """Compute deterministic SHA-256 hash of a sequence of integers."""
    payload = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Fashion-MNIST dataset splits and produce docs/split_manifest.json."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration YAML (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--recursive-config",
        type=str,
        default="configs/recursive.yaml",
        help="Path to recursive configuration YAML (default: configs/recursive.yaml).",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="docs/split_manifest.json",
        help="Destination path for split manifest (default: docs/split_manifest.json).",
    )
    return parser.parse_args(cli_args)


def verify_and_generate_manifest(
    base_cfg_path: str | Path = "configs/base.yaml",
    rec_cfg_path: str | Path = "configs/recursive.yaml",
    out_file: str | Path = "docs/split_manifest.json",
) -> dict[str, Any]:
    """Execute split verification assertions and write manifest JSON."""
    base_cfg = load_config(base_cfg_path)
    rec_cfg = load_config(rec_cfg_path)

    split_seed = int(base_cfg.data.split_seed)
    dev_size = int(base_cfg.data.dev_size)
    root = str(base_cfg.data.root)

    # Initialize data object identically to experiment scripts
    data = FashionData(root=root, split_seed=split_seed, dev_size=dev_size)

    train_indices = sorted(data.splits["train"])
    dev_indices = sorted(data.splits["dev"])
    gate_indices = sorted(list(data.get("gate").indices))
    monitor_indices = sorted(list(data.get("monitor").indices))

    # Access test set size without invoking allow_test=True
    test_size = len(data.test_dataset)
    test_indices = list(range(test_size))

    # 1. Disjointness assertions
    train_set = set(train_indices)
    dev_set = set(dev_indices)
    gate_set = set(gate_indices)
    monitor_set = set(monitor_indices)

    assert train_set.isdisjoint(dev_set), "Train and dev partitions must be strictly disjoint!"
    assert train_set | dev_set == set(range(len(data.full_train))), "Train + dev must span full 60k training set!"
    assert gate_set.isdisjoint(monitor_set), "Gate and monitor partitions must be strictly disjoint!"
    assert gate_set | monitor_set == dev_set, "Gate + monitor must partition dev set completely!"

    # 2. Phase 2 partitions for seeds 0, 1, 2
    n_labeled = int(rec_cfg.recursive.n_labeled)
    pool_size = int(rec_cfg.recursive.pool_size)
    seeds = [0, 1, 2]

    phase2_partitions: dict[str, Any] = {}
    for s in seeds:
        part = make_partition(data, n_labeled=n_labeled, pool_size=pool_size, seed=s)
        # Structural assertion: labeled, pool, gate, monitor pairwise disjoint
        assert_no_leakage(part)

        lab_sorted = sorted(part.labeled)
        pool_sorted = sorted(part.pool)

        phase2_partitions[f"seed{s}"] = {
            "labeled_size": len(lab_sorted),
            "pool_size": len(pool_sorted),
            "labeled_sha256": compute_sha256(lab_sorted),
            "pool_sha256": compute_sha256(pool_sorted),
        }

    manifest: dict[str, Any] = {
        "dataset_name": "FashionMNIST",
        "dataset_source": "torchvision.datasets.FashionMNIST",
        "split_seed": split_seed,
        "sizes": {
            "train": len(train_indices),
            "dev": len(dev_indices),
            "gate": len(gate_indices),
            "monitor": len(monitor_indices),
            "test": test_size,
        },
        "hashes": {
            "train_sha256": compute_sha256(train_indices),
            "dev_sha256": compute_sha256(dev_indices),
            "gate_sha256": compute_sha256(gate_indices),
            "monitor_sha256": compute_sha256(monitor_indices),
            "test_sha256": compute_sha256(test_indices),
        },
        "phase2_partitions": phase2_partitions,
    }

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    manifest = verify_and_generate_manifest(
        base_cfg_path=args.config,
        rec_cfg_path=args.recursive_config,
        out_file=args.out_file,
    )

    print("\n" + "=" * 70)
    print("DATASET SPLIT MANIFEST VERIFICATION")
    print("=" * 70)
    print(f"Dataset:    {manifest['dataset_name']} ({manifest['dataset_source']})")
    print(f"Split Seed: {manifest['split_seed']}")
    print("-" * 70)
    print("PARTITION SIZES & SHA-256 HASHES:")
    for k, v in manifest["sizes"].items():
        h = manifest["hashes"].get(f"{k}_sha256", "N/A")
        print(f"  {k:<10}: {v:>6} samples | SHA256: {h[:16]}...{h[-8:]}")
    print("-" * 70)
    print("PHASE 2 PARTITIONS (Seeds 0, 1, 2):")
    for s_tag, info in manifest["phase2_partitions"].items():
        print(
            f"  {s_tag}: labeled={info['labeled_size']} (hash: {info['labeled_sha256'][:8]}...), "
            f"pool={info['pool_size']} (hash: {info['pool_sha256'][:8]}...)"
        )
    print("-" * 70)
    print(f"All disjointness assertions passed successfully.")
    print(f"Manifest written to: {args.out_file}\n")


if __name__ == "__main__":
    main()
