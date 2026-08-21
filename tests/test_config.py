"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_pilot.config.resolver import load_config, merge_overrides
from ml_pilot.config.schema import TaskType


def test_load_defaults() -> None:
    cfg = load_config()
    assert cfg.split.test_size == 0.2
    assert cfg.data.task == TaskType.AUTO
    assert cfg.tune.n_trials == 40


def test_yaml_override(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("tune:\n  n_trials: 7\n", encoding="utf-8")
    cfg = load_config(config_path=path)
    assert cfg.tune.n_trials == 7


def test_cli_overrides() -> None:
    cfg = load_config(overrides={"runtime": {"compare_only": True}, "data": {"target": "y"}})
    assert cfg.runtime.compare_only is True
    assert cfg.data.target == "y"


def test_merge_overrides() -> None:
    base = load_config()
    merged = merge_overrides(base, {"save": {"output_dir": "out"}})
    assert merged.save.output_dir == "out"


def test_invalid_yaml_root(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path=path)
