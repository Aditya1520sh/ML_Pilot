"""Tests for dataset loading and task inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_pilot.config.resolver import load_config
from ml_pilot.config.schema import TaskType
from ml_pilot.core.context import PipelineContext
from ml_pilot.stages.load import LoadStage


def test_load_csv_and_infer_classification(tmp_path: Path) -> None:
    path = tmp_path / "clf.csv"
    pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": ["a", "b", "a", "b", "a", "b"],
            "label": [0, 1, 0, 1, 0, 1],
        }
    ).to_csv(path, index=False)

    cfg = load_config(overrides={"data": {"path": str(path), "target": "label"}})
    ctx = PipelineContext(config=cfg)
    ctx = LoadStage().run(ctx)
    assert ctx.task_type == TaskType.CLASSIFICATION
    assert ctx.target_name == "label"
    assert ctx.frame is not None
    assert len(ctx.frame) == 6


def test_load_infers_regression(tmp_path: Path) -> None:
    path = tmp_path / "reg.csv"
    pd.DataFrame(
        {
            "x": list(range(30)),
            "price": [float(i) * 1.5 + 3 for i in range(30)],
        }
    ).to_csv(path, index=False)

    cfg = load_config(overrides={"data": {"path": str(path), "target": "price"}})
    ctx = LoadStage().run(PipelineContext(config=cfg))
    assert ctx.task_type == TaskType.REGRESSION


def test_missing_path_raises() -> None:
    cfg = load_config()
    with pytest.raises(ValueError, match="No data path"):
        LoadStage().run(PipelineContext(config=cfg))
