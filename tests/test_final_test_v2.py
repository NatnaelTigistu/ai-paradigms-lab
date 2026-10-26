"""Unit test for Phase 1 v2 final test refusal logic."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.final_test_phase1_v2 import check_refusal


def test_final_test_v2_refuses_when_output_exists(tmp_path: Path):
    """Verify check_refusal exits with code 1 if output exists and force=False."""
    dummy_out = tmp_path / "final_test.json"
    dummy_out.write_text("{}", encoding="utf-8")

    # Verify exit code 1 when output exists without force
    with pytest.raises(SystemExit) as exc_info:
        check_refusal(dummy_out, force=False)
    assert exc_info.value.code == 1

    # Verify execution proceeds when force is set
    check_refusal(dummy_out, force=True)


def test_final_test_v2_proceeds_when_output_missing(tmp_path: Path):
    """Verify check_refusal does not exit when output file does not exist."""
    missing_out = tmp_path / "does_not_exist.json"
    check_refusal(missing_out, force=False)
