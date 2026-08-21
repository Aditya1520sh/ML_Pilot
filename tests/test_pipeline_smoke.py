"""End-to-end smoke test for PipelineRunner on a tiny synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_pilot.config.resolver import load_config
from ml_pilot.core.pipeline import PipelineRunner


def _make_classification_csv(path: Path, n: int = 80) -> None:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["red", "blue", "green"], size=n)
    logits = 1.2 * x1 - 0.8 * x2 + (cat == "red") * 0.5
    y = (logits > 0).astype(int)
    pd.DataFrame({"x1": x1, "x2": x2, "color": cat, "target": y}).to_csv(path, index=False)


def test_pipeline_compare_only(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    out_dir = tmp_path / "artifacts"
    _make_classification_csv(data_path)

    cfg = load_config(
        overrides={
            "data": {"path": str(data_path), "target": "target", "task": "classification"},
            "save": {"output_dir": str(out_dir)},
            "runtime": {"compare_only": True, "no_shap": True, "verbose": False},
            "compare": {
                "cv_folds": 2,
                "n_jobs": 1,
                "include_models": ["logistic_regression", "decision_tree", "random_forest"],
            },
            "tune": {"enable": False},
        }
    )
    runner = PipelineRunner(config=cfg)
    ctx = runner.run()

    assert ctx.best_model_name is not None
    assert ctx.comparison_table is not None
    assert len(ctx.comparison_table) >= 1
    assert (out_dir / "mlpilot_pipeline.joblib").exists()
    assert (out_dir / "run_metadata.json").exists()


def test_registry_order() -> None:
    from ml_pilot.core.defaults import build_default_registry

    registry = build_default_registry()
    names = registry.names()
    assert names[0] == "load"
    assert names[-1] == "deploy"
    assert "compare" in names
    assert "tune" in names
