"""Explain stage — SHAP summary, dependence, and waterfall plots.

Purpose:
    Produce model explanations for stakeholders and debugging.

Interactions:
    - Uses fitted model and train features.
    - Writes plot artifacts under the output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage
from ml_pilot.stages.preprocess import transform_features
from ml_pilot.utils.io import ensure_directory


class ExplainStage(BaseStage):
    """Generate SHAP explanation figures."""

    @property
    def name(self) -> str:
        return "explain"

    def should_skip(self, context: PipelineContext) -> bool:
        if context.config.runtime.compare_only:
            return True
        if context.config.runtime.no_shap:
            return True
        return not context.config.explain.enable

    def run(self, context: PipelineContext) -> PipelineContext:
        """Create SHAP plots and register artifact paths.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        X_train, _, _, _ = context.require_splits()
        if context.model is None or context.preprocessor is None:
            raise RuntimeError("Model/preprocessor required for explain stage.")

        cfg = context.config.explain
        out_dir = ensure_directory(Path(context.config.save.output_dir) / "shap")
        X_tr = transform_features(context.preprocessor, X_train)
        sample_n = min(len(X_tr), cfg.max_samples)
        sample = X_tr.sample(n=sample_n, random_state=context.config.runtime.seed)
        estimator = context.model.named_steps["model"]

        try:
            explainer = shap.Explainer(estimator, sample)
            shap_values = explainer(sample)
        except Exception as exc:  # noqa: BLE001
            context.add_warning(f"Explain stage skipped: SHAP explainer failed ({exc}).")
            return context

        context.extras["shap_values"] = shap_values
        plots = set(cfg.plots)

        if "summary" in plots:
            path = out_dir / "shap_summary.png"
            plt.figure()
            shap.summary_plot(shap_values, sample, show=False)
            plt.tight_layout()
            plt.savefig(path, dpi=140, bbox_inches="tight")
            plt.close()
            context.store_artifact("shap_summary", str(path))

        if "dependence" in plots:
            feature = self._top_feature(context, sample)
            path = out_dir / f"shap_dependence_{feature}.png"
            plt.figure()
            try:
                shap.dependence_plot(feature, self._values_2d(shap_values), sample, show=False)
                plt.tight_layout()
                plt.savefig(path, dpi=140, bbox_inches="tight")
                context.store_artifact("shap_dependence", str(path))
            except Exception as exc:  # noqa: BLE001
                context.add_warning(f"Dependence plot failed: {exc}")
            finally:
                plt.close()

        if "waterfall" in plots:
            path = out_dir / "shap_waterfall.png"
            plt.figure()
            try:
                # Waterfall for the first row explanation.
                if hasattr(shap_values, "__getitem__"):
                    shap.plots.waterfall(shap_values[0], show=False)
                else:
                    shap.plots.waterfall(shap_values, show=False)
                plt.tight_layout()
                plt.savefig(path, dpi=140, bbox_inches="tight")
                context.store_artifact("shap_waterfall", str(path))
            except Exception as exc:  # noqa: BLE001
                context.add_warning(f"Waterfall plot failed: {exc}")
            finally:
                plt.close()

        return context

    @staticmethod
    def _values_2d(shap_values: Any) -> np.ndarray:
        """Normalize SHAP values to a 2D array for dependence plots."""
        raw = getattr(shap_values, "values", shap_values)
        if isinstance(raw, list):
            raw = raw[0]
        arr = np.asarray(raw)
        if arr.ndim == 3:
            return arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
        return arr

    @staticmethod
    def _top_feature(context: PipelineContext, sample: pd.DataFrame) -> str:
        """Pick the highest-importance feature name when available."""
        if context.feature_importances is not None and not context.feature_importances.empty:
            return str(context.feature_importances.iloc[0]["feature"])
        return str(sample.columns[0])
