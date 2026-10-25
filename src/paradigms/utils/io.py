"""Input/output utilities for serializing and deserializing project data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save serializable data to a JSON file.

    Ensures the parent directory exists before writing to prevent file-not-found
    errors when saving newly generated artifacts or splits.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str | Path) -> Any:
    """Load and parse JSON data from a file path."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
