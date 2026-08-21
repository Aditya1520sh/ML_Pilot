"""Shared pipeline state container.

Purpose:
    Hold all mutable artefacts produced while stages execute so later stages
    can read earlier results without tight coupling.

Interactions:
    - Created by ``PipelineRunner`` and passed into every ``BaseStage.run``.
    - Written by stages (frames, models, metrics, artifact paths).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ml_pilot.config.schema import MLPilotConfig, TaskType


@dataclass
class PipelineContext:
    """Mutable shared state for a single MLPilot run.

    Attributes:
        config: Resolved run configuration.
        raw_frame: Original loaded table.
        frame: Working dataframe after clean / feature engineering.
        target_name: Column used as the learning target.
        task_type: Resolved regression or classification task.
        feature_names: Feature columns after engineering (pre-transform).
        X_train: Training features (engineered, pre-ColumnTransformer).
        X_test: Hold-out features.
        y_train: Training labels.
        y_test: Hold-out labels.
        preprocessor: Fitted sklearn ColumnTransformer / Pipeline.
        model: Best estimator (possibly wrapped).
        inference_bundle: Object exported for serving (feature eng + model).
        comparison_table: Model comparison leaderboard.
        best_model_name: Name of the winning model family.
        metrics: Evaluation metric dictionary.
        feature_importances: Ranked feature importance table.
        artifacts: Mapping of artifact keys to filesystem paths.
        stage_timings: Wall-clock seconds per stage name.
        warnings: Human-readable warnings accumulated during the run.
        extras: Escape hatch for stage-specific payloads.
    """

    config: MLPilotConfig
    raw_frame: pd.DataFrame | None = None
    frame: pd.DataFrame | None = None
    target_name: str | None = None
    task_type: TaskType = TaskType.AUTO
    feature_names: list[str] = field(default_factory=list)
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None
    preprocessor: Any = None
    model: Any = None
    inference_bundle: Any = None
    comparison_table: pd.DataFrame | None = None
    best_model_name: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_importances: pd.DataFrame | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def require_frame(self) -> pd.DataFrame:
        """Return the working frame or raise if missing."""
        if self.frame is None:
            raise RuntimeError("PipelineContext.frame is not set. Run Load/Clean first.")
        return self.frame

    def require_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return train/test matrices or raise if missing."""
        if self.X_train is None or self.X_test is None:
            raise RuntimeError("Train/test features missing. Run Split first.")
        if self.y_train is None or self.y_test is None:
            raise RuntimeError("Train/test targets missing. Run Split first.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def add_warning(self, message: str) -> None:
        """Append a warning string for later display."""
        self.warnings.append(message)

    def store_artifact(self, key: str, path: str) -> None:
        """Record an artifact path under ``key``."""
        self.artifacts[key] = path
