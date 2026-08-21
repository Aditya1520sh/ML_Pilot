"""Save stage — export inference bundle and run metadata.

Purpose:
    Persist feature engineering + preprocessing + model as a joblib artifact
    plus a JSON metadata sidecar.

Interactions:
    - Reads engineer, model, metrics, config from context.
    - Writes filesystem artifacts referenced in ``context.artifacts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage
from ml_pilot.utils.io import ensure_directory, write_json


@dataclass
class InferenceBundle:
    """Serializable object used for offline / batch inference.

    Attributes:
        feature_engineer: Fitted ``TabularFeatureEngineer`` (optional).
        model_pipeline: Sklearn pipeline with preprocess + estimator.
        target_name: Name of the prediction target.
        task_type: Resolved task string.
        feature_names: Raw engineered feature columns expected before preprocess.
    """

    feature_engineer: Any
    model_pipeline: Any
    target_name: str | None
    task_type: str
    feature_names: list[str]

    def predict(self, frame: pd.DataFrame) -> Any:
        """Run feature engineering (if present) then model prediction.

        Args:
            frame: Raw feature dataframe (no target column required).

        Returns:
            Model predictions.
        """
        features = frame.copy()
        if self.target_name and self.target_name in features.columns:
            features = features.drop(columns=[self.target_name])
        if self.feature_engineer is not None:
            features = self.feature_engineer.transform(features)
        return self.model_pipeline.predict(features)


class SaveStage(BaseStage):
    """Export the full inference bundle and metadata."""

    @property
    def name(self) -> str:
        return "save"

    def run(self, context: PipelineContext) -> PipelineContext:
        """Write joblib + JSON artifacts to the configured output directory.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        if context.model is None:
            raise RuntimeError("Cannot save: model is missing.")

        out_dir = ensure_directory(context.config.save.output_dir)
        bundle = InferenceBundle(
            feature_engineer=context.extras.get("feature_engineer"),
            model_pipeline=context.model,
            target_name=context.target_name,
            task_type=context.task_type.value
            if hasattr(context.task_type, "value")
            else str(context.task_type),
            feature_names=list(context.feature_names),
        )
        context.inference_bundle = bundle

        model_path = Path(out_dir) / context.config.save.model_filename
        joblib.dump(bundle, model_path)
        context.store_artifact("model_bundle", str(model_path))

        metadata = {
            "best_model_name": context.best_model_name,
            "task_type": bundle.task_type,
            "target_name": context.target_name,
            "feature_names": context.feature_names,
            "metrics": context.metrics,
            "stage_timings": context.stage_timings,
            "artifacts": context.artifacts,
            "config": context.config.model_dump_plain(),
            "warnings": context.warnings,
        }
        meta_path = write_json(Path(out_dir) / context.config.save.metadata_filename, metadata)
        context.store_artifact("metadata", str(meta_path))

        if context.comparison_table is not None:
            csv_path = Path(out_dir) / "leaderboard.csv"
            context.comparison_table.to_csv(csv_path, index=False)
            context.store_artifact("leaderboard_csv", str(csv_path))

        return context
