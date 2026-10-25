"""Configuration loading, dot-attribute access, and override utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dictionary subclass providing attribute-style dot access to nested keys.

    Allows accessing nested configurations via either `cfg['training']['epochs']`
    or `cfg.training.epochs` seamlessly.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
        except KeyError:
            raise AttributeError(f"Config object has no attribute '{name}'")

        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
            self[name] = val
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Config object has no attribute '{name}'")


def load_config(path: str | Path) -> Config:
    """Load and parse a YAML configuration file into a Config object.

    Args:
        path: Path to the .yaml configuration file.

    Returns:
        Config instance with dot-accessible keys.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return _to_config(data)


def _to_config(obj: Any) -> Any:
    """Recursively convert nested dictionaries to Config objects."""
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_config(item) for item in obj]
    return obj


def apply_overrides(cfg: Config | dict[str, Any], overrides: dict[str, Any]) -> Config:
    """Apply dictionary or dot-notated key overrides to a configuration.

    Supports:
        - Nested dictionary overrides: `{"training": {"epochs": 5}}`
        - Dot-notated string keys: `{"training.epochs": 5}`
        - Top-level shorthand: `{"epochs": 5}` (mapped to `cfg.training.epochs` if present)

    Args:
        cfg: Base configuration object.
        overrides: Mapping of keys to override values.

    Returns:
        New Config instance reflecting the applied overrides.
    """
    new_cfg = _to_config(dict(cfg))

    for key, value in overrides.items():
        if value is None:
            continue

        if "." in key:
            parts = key.split(".")
            curr = new_cfg
            for part in parts[:-1]:
                if part not in curr or not isinstance(curr[part], dict):
                    curr[part] = Config()
                curr = curr[part]
            curr[parts[-1]] = _to_config(value)
        elif key in new_cfg and isinstance(new_cfg[key], dict) and isinstance(value, dict):
            new_cfg[key] = apply_overrides(new_cfg[key], value)
        elif key in new_cfg:
            new_cfg[key] = _to_config(value)
        elif "training" in new_cfg and isinstance(new_cfg["training"], dict) and key in new_cfg["training"]:
            new_cfg["training"][key] = _to_config(value)
        else:
            new_cfg[key] = _to_config(value)

    return new_cfg
