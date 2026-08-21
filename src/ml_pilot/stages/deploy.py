"""Deploy stage — write a lightweight serving checklist / smoke-test script.

Purpose:
    Emit deployment helpers next to the saved bundle (load snippet + schema).
    This stage does not start a server; it prepares artifacts for serving.

Interactions:
    - Depends on Save stage outputs.
    - Adds ``deploy_readme`` / ``predict_snippet`` artifact paths.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage
from ml_pilot.utils.io import ensure_directory, write_json


class DeployStage(BaseStage):
    """Prepare deployment artifacts for the exported inference bundle."""

    @property
    def name(self) -> str:
        return "deploy"

    def should_skip(self, context: PipelineContext) -> bool:
        return context.config.runtime.compare_only

    def run(self, context: PipelineContext) -> PipelineContext:
        """Write predict helper and input schema beside the model.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        out_dir = ensure_directory(context.config.save.output_dir)
        model_name = context.config.save.model_filename

        schema = {
            "target_name": context.target_name,
            "task_type": context.task_type.value
            if hasattr(context.task_type, "value")
            else str(context.task_type),
            "feature_names": context.feature_names,
            "model_file": model_name,
        }
        schema_path = write_json(Path(out_dir) / "serving_schema.json", schema)
        context.store_artifact("serving_schema", str(schema_path))

        snippet = dedent(
            f"""
            # MLPilot inference smoke test
            import joblib
            import pandas as pd

            bundle = joblib.load({model_name!r})
            # frame = pd.read_csv("new_data.csv")
            # preds = bundle.predict(frame)
            # print(preds[:5])
            """
        ).strip()
        snippet_path = Path(out_dir) / "predict_snippet.py"
        snippet_path.write_text(snippet + "\n", encoding="utf-8")
        context.store_artifact("predict_snippet", str(snippet_path))

        readme = dedent(
            f"""
            # MLPilot Deploy Notes

            - Load bundle: `joblib.load("{model_name}")`
            - Call `bundle.predict(dataframe)` with columns matching training features
              (target column optional; dropped automatically if present).
            - See `serving_schema.json` for expected feature names and task type.
            """
        ).strip()
        readme_path = Path(out_dir) / "DEPLOY.md"
        readme_path.write_text(readme + "\n", encoding="utf-8")
        context.store_artifact("deploy_readme", str(readme_path))
        return context
