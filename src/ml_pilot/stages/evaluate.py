"""Evaluate stage — hold-out metrics and overfitting checks.

Purpose:
    Score the final model on train and test splits with task-appropriate metrics.

Interactions:
    - Reads ``context.model`` and splits.
    - Writes ``context.metrics``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
)

from ml_pilot.config.schema import TaskType
from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage
from ml_pilot.utils.io import ensure_directory, write_json


class EvaluateStage(BaseStage):
    """Compute evaluation metrics and detect train/test gaps."""

    @property
    def name(self) -> str:
        return "evaluate"

    def should_skip(self, context: PipelineContext) -> bool:
        return context.config.runtime.compare_only

    def run(self, context: PipelineContext) -> PipelineContext:
        """Evaluate the fitted inference pipeline on hold-out data.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        X_train, X_test, y_train, y_test = context.require_splits()
        if context.model is None:
            raise RuntimeError("No model available for evaluation.")

        y_train_pred = context.model.predict(X_train)
        y_test_pred = context.model.predict(X_test)

        if context.task_type == TaskType.CLASSIFICATION:
            metrics = self._classification_metrics(
                y_train, y_test, y_train_pred, y_test_pred, context
            )
            primary_train = metrics.get("train_f1", 0.0)
            primary_test = metrics.get("test_f1", 0.0)
        else:
            metrics = self._regression_metrics(y_train, y_test, y_train_pred, y_test_pred)
            primary_train = metrics.get("train_r2", 0.0)
            primary_test = metrics.get("test_r2", 0.0)

        gap = float(primary_train - primary_test)
        metrics["overfit_gap"] = gap
        threshold = context.config.evaluate.overfit_gap_threshold
        metrics["overfitting_flag"] = bool(gap > threshold)
        if metrics["overfitting_flag"]:
            context.add_warning(
                f"Possible overfitting: train-test gap={gap:.4f} exceeds {threshold}."
            )

        context.metrics = metrics
        out_dir = ensure_directory(context.config.save.output_dir)
        path = write_json(out_dir / "metrics.json", metrics)
        context.store_artifact("metrics", str(path))
        return context

    def _classification_metrics(
        self,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Build classification metric dictionary."""
        metrics: dict[str, Any] = {
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "train_f1": float(f1_score(y_train, y_train_pred, average="weighted", zero_division=0)),
            "test_f1": float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0)),
            "train_mcc": float(matthews_corrcoef(y_train, y_train_pred)),
            "test_mcc": float(matthews_corrcoef(y_test, y_test_pred)),
        }
        try:
            if hasattr(context.model, "predict_proba"):
                proba_test = context.model.predict_proba(context.X_test)
                if proba_test.shape[1] == 2:
                    metrics["test_roc_auc"] = float(roc_auc_score(y_test, proba_test[:, 1]))
                else:
                    metrics["test_roc_auc"] = float(
                        roc_auc_score(y_test, proba_test, multi_class="ovr", average="weighted")
                    )
        except Exception as exc:  # noqa: BLE001
            context.add_warning(f"ROC-AUC unavailable: {exc}")
        return metrics

    @staticmethod
    def _regression_metrics(y_train, y_test, y_train_pred, y_test_pred) -> dict[str, Any]:
        """Build regression metric dictionary."""
        def _mape(y_true, y_pred) -> float:
            try:
                return float(mean_absolute_percentage_error(y_true, y_pred))
            except Exception:
                return float("nan")

        return {
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "test_r2": float(r2_score(y_test, y_test_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
            "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
            "train_mape": _mape(y_train, y_train_pred),
            "test_mape": _mape(y_test, y_test_pred),
        }
