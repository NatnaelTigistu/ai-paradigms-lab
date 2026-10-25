"""Tests for configuration parsing, dot-attribute access, and override logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from paradigms.config import Config, apply_overrides, load_config


def test_config_dot_and_dict_access() -> None:
    """Verify Config allows both dictionary key and attribute dot access."""
    cfg = Config({"data": {"batch_size": 64}, "epochs": 5})

    assert cfg["epochs"] == 5
    assert cfg.epochs == 5
    assert cfg["data"]["batch_size"] == 64
    assert cfg.data.batch_size == 64


def test_config_missing_attribute_raises() -> None:
    """Verify accessing nonexistent attribute raises AttributeError."""
    cfg = Config({"epochs": 5})
    with pytest.raises(AttributeError, match="Config object has no attribute 'missing'"):
        _ = cfg.missing


def test_apply_overrides_nested_and_dot_notation() -> None:
    """Verify apply_overrides handles nested dicts, dot strings, and shorthand keys."""
    base = Config({
        "data": {"root": "./data", "batch_size": 128},
        "training": {"epochs": 10, "lr": 0.001},
        "seeds": [0, 1, 2],
    })

    # Test dot-notation override
    overridden = apply_overrides(base, {"training.epochs": 20})
    assert overridden.training.epochs == 20
    assert overridden.training.lr == 0.001

    # Test shorthand override mapping to training.epochs
    shorthand = apply_overrides(base, {"epochs": 15})
    assert shorthand.training.epochs == 15

    # Test new key addition
    added = apply_overrides(base, {"verbose": True})
    assert added.verbose is True


def test_load_config_from_file(tmp_path: Path) -> None:
    """Verify loading YAML configuration from file."""
    yaml_content = """
data:
  root: "./custom_data"
training:
  epochs: 7
  lr: 0.005
"""
    p = tmp_path / "test_cfg.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    loaded = load_config(p)
    assert loaded.data.root == "./custom_data"
    assert loaded.training.epochs == 7
    assert loaded.training.lr == 0.005


def test_load_config_missing_file_raises() -> None:
    """Verify load_config raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config_file.yaml")
