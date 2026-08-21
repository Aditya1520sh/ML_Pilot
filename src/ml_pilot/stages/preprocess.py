"""Preprocess stage — ColumnTransformer for impute / scale / encode.

Purpose:
    Build and fit a sklearn preprocessing pipeline on training features only.

Interactions:
    - Fits on ``X_train``; stores transformer on ``context.preprocessor``.
    - Used by Compare, Tune, Evaluate, Explain, and Save.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage


class PreprocessStage(BaseStage):
    """Construct and fit the preprocessing ColumnTransformer."""

    @property
    def name(self) -> str:
        return "preprocess"

    def run(self, context: PipelineContext) -> PipelineContext:
        """Fit preprocessor on training features.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        X_train, _, _, _ = context.require_splits()
        cfg = context.config.preprocess

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

        transformers: list[tuple[str, Pipeline, list[str]]] = []

        if numeric_cols:
            num_steps: list[tuple[str, object]] = [
                ("imputer", SimpleImputer(strategy=cfg.numeric_impute)),
            ]
            scaler = self._build_scaler(cfg.scaler)
            if scaler is not None:
                num_steps.append(("scaler", scaler))
            transformers.append(("numeric", Pipeline(num_steps), numeric_cols))

        if categorical_cols:
            encoder: object
            if cfg.encoder == "ordinal":
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            else:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=cfg.categorical_impute, fill_value="missing")),
                    ("encoder", encoder),
                ]
            )
            transformers.append(("categorical", cat_pipe, categorical_cols))

        if not transformers:
            raise ValueError("No features available for preprocessing.")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        preprocessor.set_output(transform="pandas")
        preprocessor.fit(X_train)

        context.preprocessor = preprocessor
        context.extras["numeric_cols"] = numeric_cols
        context.extras["categorical_cols"] = categorical_cols
        return context

    @staticmethod
    def _build_scaler(name: str):
        """Return a scaler instance or None."""
        mapping = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "none": None,
        }
        if name not in mapping:
            raise ValueError(f"Unknown scaler '{name}'")
        return mapping[name]


def transform_features(preprocessor: ColumnTransformer, frame: pd.DataFrame) -> pd.DataFrame:
    """Transform a feature frame with a fitted preprocessor to a dense DataFrame."""
    transformed = preprocessor.transform(frame)
    if isinstance(transformed, pd.DataFrame):
        return transformed
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=names, index=frame.index)
