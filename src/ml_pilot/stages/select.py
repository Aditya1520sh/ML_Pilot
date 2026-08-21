"""Select stage — SHAP / permutation feature importance ranking.

Purpose:
    Rank transformed features and optionally prune low-importance columns.

Interactions:
    - Uses trained model + preprocessor.
    - Writes ``feature_importances`` and may refit a reduced pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage
from ml_pilot.stages.preprocess import transform_features
from ml_pilot.utils.io import ensure_directory, write_json


class SelectStage(BaseStage):
    """Feature importance ranking and optional top-k retention."""

    @property
    def name(self) -> str:
        return "select"

    def should_skip(self, context: PipelineContext) -> bool:
        if not context.config.select.enable:
            return True
        if context.config.runtime.no_shap and context.config.select.importance_method == "shap":
            return True
        return context.config.runtime.compare_only

    def run(self, context: PipelineContext) -> PipelineContext:
        """Compute importances and optionally shrink the feature set.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        X_train, X_test, y_train, _ = context.require_splits()
        if context.model is None or context.preprocessor is None:
            raise RuntimeError("Model/preprocessor required for select stage.")

        cfg = context.config.select
        X_tr = transform_features(context.preprocessor, X_train)
        estimator = context.model.named_steps["model"]

        if cfg.importance_method == "permutation":
            result = permutation_importance(
                estimator,
                X_tr,
                y_train,
                n_repeats=5,
                random_state=context.config.runtime.seed,
                n_jobs=context.config.compare.n_jobs,
            )
            importances = result.importances_mean
        else:
            importances = self._shap_importances(estimator, X_tr, context)

        table = pd.DataFrame(
            {
                "feature": list(X_tr.columns),
                "importance": np.abs(importances),
            }
        ).sort_values("importance", ascending=False)
        total = float(table["importance"].sum()) or 1.0
        table["share"] = table["importance"] / total
        context.feature_importances = table.reset_index(drop=True)

        keep = self._choose_features(table, cfg.top_k, cfg.min_importance_share)
        context.extras["selected_transformed_features"] = keep

        out_dir = ensure_directory(context.config.save.output_dir)
        path = write_json(
            out_dir / "feature_importances.json",
            table.to_dict(orient="records"),
        )
        context.store_artifact("feature_importances", str(path))

        # Note: selection ranking is informational; full pipeline retained for
        # stability unless top_k explicitly shrinks transformed space via mask.
        if cfg.top_k is not None and cfg.top_k < len(keep):
            context.add_warning(
                "top_k set but raw-column pruning is skipped to keep "
                "ColumnTransformer schema intact; rankings are exported for review."
            )
        return context

    def _shap_importances(
        self,
        estimator: Any,
        X_tr: pd.DataFrame,
        context: PipelineContext,
    ) -> np.ndarray:
        """Mean |SHAP| per feature with a safe explainer fallback."""
        sample_n = min(len(X_tr), context.config.explain.max_samples)
        sample = X_tr.sample(n=sample_n, random_state=context.config.runtime.seed)
        try:
            explainer = shap.Explainer(estimator, sample)
            values = explainer(sample)
            raw = values.values
            if isinstance(raw, list):
                raw = raw[0]
            arr = np.asarray(raw)
            if arr.ndim == 3:
                arr = arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
            return np.mean(np.abs(arr), axis=0)
        except Exception as exc:  # noqa: BLE001
            context.add_warning(f"SHAP failed ({exc}); falling back to permutation importance.")
            _, _, y_train, _ = context.require_splits()
            y_sample = y_train.loc[sample.index]
            result = permutation_importance(
                estimator,
                sample,
                y_sample,
                n_repeats=3,
                random_state=context.config.runtime.seed,
            )
            return result.importances_mean

    @staticmethod
    def _choose_features(
        table: pd.DataFrame,
        top_k: int | None,
        min_share: float,
    ) -> list[str]:
        """Select feature names by top-k and/or cumulative share rules."""
        ordered = table.sort_values("importance", ascending=False)
        if top_k is not None:
            ordered = ordered.head(top_k)
        if min_share > 0:
            ordered = ordered[ordered["share"] >= min_share]
        return ordered["feature"].tolist()
