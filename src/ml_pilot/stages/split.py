"""Split stage — stratified train/test partition.

Purpose:
    Create reproducible train/test matrices from the engineered frame.

Interactions:
    - Writes ``X_train``, ``X_test``, ``y_train``, ``y_test`` on context.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from ml_pilot.config.schema import TaskType
from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage


class SplitStage(BaseStage):
    """Train/test splitting with optional stratification."""

    @property
    def name(self) -> str:
        return "split"

    def run(self, context: PipelineContext) -> PipelineContext:
        """Split features and target into train/test sets.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        frame = context.require_frame()
        target = context.target_name
        if not target or target not in frame.columns:
            raise ValueError("Target column missing; cannot split.")

        cfg = context.config.split
        X = frame.drop(columns=[target])
        y = frame[target]

        stratify = None
        if cfg.stratify and context.task_type == TaskType.CLASSIFICATION:
            # Stratify only when every class has enough members.
            counts = y.value_counts()
            if counts.min() >= 2:
                stratify = y
            else:
                context.add_warning("Stratify disabled: some classes have < 2 samples.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify,
        )

        context.X_train = X_train.reset_index(drop=True)
        context.X_test = X_test.reset_index(drop=True)
        context.y_train = y_train.reset_index(drop=True)
        context.y_test = y_test.reset_index(drop=True)
        context.feature_names = list(X.columns)
        return context
