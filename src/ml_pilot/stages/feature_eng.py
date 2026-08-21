"""Feature engineering stage — datetime, logs, flags, rare categories.

Purpose:
    Expand the feature space with deterministic, fit-free transforms that
    remain reproducible at inference time via a stored transformer object.

Interactions:
    - Updates ``context.frame`` and ``feature_names``.
    - Stores ``extras['feature_engineer']`` for the Save/Deploy bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage


@dataclass
class FeatureEngineerState:
    """Serializable parameters learned during fit of ``TabularFeatureEngineer``."""

    datetime_columns: list[str] = field(default_factory=list)
    log_columns: list[str] = field(default_factory=list)
    missing_flag_columns: list[str] = field(default_factory=list)
    rare_maps: dict[str, set[str]] = field(default_factory=dict)
    interaction_pairs: list[tuple[str, str]] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)


class TabularFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible feature engineer for tabular frames."""

    def __init__(
        self,
        target_name: str | None = None,
        datetime_decompose: bool = True,
        log_skew_threshold: float = 1.0,
        add_missingness_flags: bool = True,
        pairwise_interactions: bool = False,
        rare_category_threshold: float = 0.01,
    ) -> None:
        self.target_name = target_name
        self.datetime_decompose = datetime_decompose
        self.log_skew_threshold = log_skew_threshold
        self.add_missingness_flags = add_missingness_flags
        self.pairwise_interactions = pairwise_interactions
        self.rare_category_threshold = rare_category_threshold
        self.state_: FeatureEngineerState | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TabularFeatureEngineer:
        """Learn which transforms to apply from training data only."""
        frame = X.copy()
        state = FeatureEngineerState()

        # Detect datetime-like columns.
        for col in frame.columns:
            if col == self.target_name:
                continue
            if pd.api.types.is_datetime64_any_dtype(frame[col]):
                state.datetime_columns.append(col)
            elif frame[col].dtype == object:
                parsed = pd.to_datetime(frame[col], errors="coerce", utc=False)
                if parsed.notna().mean() > 0.8:
                    state.datetime_columns.append(col)

        numeric_cols = [
            c
            for c in frame.select_dtypes(include=[np.number]).columns
            if c != self.target_name and c not in state.datetime_columns
        ]
        for col in numeric_cols:
            skew = float(frame[col].dropna().skew()) if frame[col].notna().any() else 0.0
            if abs(skew) >= self.log_skew_threshold and (frame[col].dropna() >= 0).all():
                state.log_columns.append(col)

        if self.add_missingness_flags:
            state.missing_flag_columns = [
                c for c in frame.columns if c != self.target_name and frame[c].isna().any()
            ]

        cat_cols = [
            c
            for c in frame.columns
            if c != self.target_name
            and c not in state.datetime_columns
            and not pd.api.types.is_numeric_dtype(frame[c])
        ]
        for col in cat_cols:
            freq = frame[col].astype("string").value_counts(normalize=True)
            rares = set(freq[freq < self.rare_category_threshold].index.astype(str))
            if rares:
                state.rare_maps[col] = rares

        if self.pairwise_interactions and len(numeric_cols) >= 2:
            # Limit to first few numeric columns to avoid combinatorial blow-up.
            limited = numeric_cols[:5]
            state.interaction_pairs = [
                (limited[i], limited[j])
                for i in range(len(limited))
                for j in range(i + 1, len(limited))
            ]

        # Probe transform to capture output schema.
        self.state_ = state
        transformed = self.transform(frame)
        state.output_columns = list(transformed.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted feature engineering to a dataframe."""
        if self.state_ is None:
            raise RuntimeError("TabularFeatureEngineer must be fitted before transform.")
        state = self.state_
        frame = X.copy()

        for col in state.datetime_columns:
            if col not in frame.columns:
                continue
            values = pd.to_datetime(frame[col], errors="coerce")
            frame[f"{col}__year"] = values.dt.year
            frame[f"{col}__month"] = values.dt.month
            frame[f"{col}__day"] = values.dt.day
            frame[f"{col}__dow"] = values.dt.dayofweek
            frame = frame.drop(columns=[col])

        for col in state.log_columns:
            if col in frame.columns:
                frame[f"{col}__log1p"] = np.log1p(frame[col].clip(lower=0))

        for col in state.missing_flag_columns:
            if col in frame.columns:
                frame[f"{col}__was_missing"] = frame[col].isna().astype(int)

        for col, rares in state.rare_maps.items():
            if col in frame.columns:
                as_str = frame[col].astype("string")
                frame[col] = as_str.where(~as_str.isin(rares), other="__rare__")

        for left, right in state.interaction_pairs:
            if left in frame.columns and right in frame.columns:
                frame[f"{left}__x__{right}"] = frame[left] * frame[right]

        if self.target_name and self.target_name in frame.columns:
            # Keep target if present; caller may drop later.
            pass
        return frame


class FeatureEngStage(BaseStage):
    """Apply tabular feature engineering to the working frame."""

    @property
    def name(self) -> str:
        return "feature_eng"

    def run(self, context: PipelineContext) -> PipelineContext:
        """Fit engineer on the full cleaned frame and transform it.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context with engineered features.
        """
        frame = context.require_frame()
        target = context.target_name
        cfg = context.config.feature_eng

        engineer = TabularFeatureEngineer(
            target_name=target,
            datetime_decompose=cfg.datetime_decompose,
            log_skew_threshold=cfg.log_skew_threshold,
            add_missingness_flags=cfg.add_missingness_flags,
            pairwise_interactions=cfg.pairwise_interactions,
            rare_category_threshold=cfg.rare_category_threshold,
        )

        feature_frame = frame.drop(columns=[target]) if target and target in frame.columns else frame
        engineer.fit(feature_frame)
        transformed = engineer.transform(feature_frame)
        if target and target in frame.columns:
            transformed[target] = frame[target].values

        context.frame = transformed
        context.feature_names = [c for c in transformed.columns if c != target]
        context.extras["feature_engineer"] = engineer
        return context
