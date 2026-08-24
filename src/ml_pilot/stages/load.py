"""Load stage — ingest tabular files and infer task type.

Purpose:
    Read CSV/Parquet/Excel, resolve the target column, and classify the task
    as regression or classification.

Interactions:
    - Uses ``utils.io.read_tabular``.
    - Writes ``raw_frame``, ``frame``, ``target_name``, ``task_type`` on context.
"""

from __future__ import annotations

import pandas as pd

from ml_pilot.config.schema import TaskType
from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage
from ml_pilot.utils.io import read_tabular


class LoadStage(BaseStage):
    """Load dataset and infer learning task."""

    @property
    def name(self) -> str:
        return "load"

    def run(self, context: PipelineContext) -> PipelineContext:
        """Load data from disk and populate context fields.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        path = context.config.data.path
        if not path:
            raise ValueError("No data path provided. Pass --data or set data.path in config.")

        # Detect duplicate headers before pandas renames them
        raw_cols = pd.read_csv(path, nrows=0).columns
        dup_cols = raw_cols[raw_cols.duplicated()]

        if len(dup_cols):
            raise ValueError(
                f"Duplicate column names detected: {dup_cols.tolist()}"
            )

        frame = read_tabular(path)
        if frame.empty:
            raise ValueError(f"Loaded dataset is empty: {path}")

        if len(frame) < 2:
            raise ValueError(f"Dataset contains only {len(frame)} row. Minimum 2 rows are required.")

        target = context.config.data.target or self._guess_target(frame)
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' not found in columns: {list(frame.columns)}")

        drop_cols = [
            col
            for col in context.config.data.drop_columns + context.config.data.id_columns
            if col in frame.columns and col != target
        ]
        if drop_cols:
            frame = frame.drop(columns=drop_cols)

        task = context.config.data.task
        if task == TaskType.AUTO:
            task = self._infer_task(frame[target])

        context.raw_frame = frame.copy()
        context.frame = frame
        context.target_name = target
        context.task_type = task
        context.extras["n_rows"] = int(len(frame))
        context.extras["n_cols"] = int(frame.shape[1])
        return context

    @staticmethod
    def _guess_target(frame: pd.DataFrame) -> str:
        """Heuristic: prefer columns named like target/label, else last column."""
        candidates = ("target", "label", "y", "class", "outcome", "response")
        lowered = {col.lower(): col for col in frame.columns}
        for name in candidates:
            if name in lowered:
                return lowered[name]
        return str(frame.columns[-1])

    @staticmethod
    def _infer_task(series: pd.Series) -> TaskType:
        """Infer regression vs classification from the target series."""
        non_null = series.dropna()
        if non_null.empty:
            raise ValueError("Target column contains only missing values.")

        if pd.api.types.is_bool_dtype(non_null):
            return TaskType.CLASSIFICATION
        if pd.api.types.is_numeric_dtype(non_null):
            n_unique = int(non_null.nunique())
            # Small integer-like cardinality → classification.
            if n_unique <= max(10, int(0.05 * len(non_null))):
                return TaskType.CLASSIFICATION
            return TaskType.REGRESSION
        return TaskType.CLASSIFICATION
