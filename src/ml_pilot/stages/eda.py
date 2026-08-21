"""EDA stage — profile the working dataframe.

Purpose:
    Compute missingness, duplicates, dtypes, correlations, imbalance, and
    summary statistics; store a report on the context.

Interactions:
    - Reads ``context.frame`` / ``target_name``.
    - Writes ``extras['eda_report']`` and optional artifact HTML/JSON.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_pilot.config.schema import TaskType
from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage
from ml_pilot.utils.io import ensure_directory, write_json


class EdaStage(BaseStage):
    """Exploratory data analysis and profiling."""

    @property
    def name(self) -> str:
        return "eda"

    def run(self, context: PipelineContext) -> PipelineContext:
        """Build an EDA report and attach it to the context.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        frame = context.require_frame()
        target = context.target_name
        report: dict[str, Any] = {
            "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
            "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
            "missing": {
                str(col): float(ratio)
                for col, ratio in frame.isna().mean().sort_values(ascending=False).round(4).items()
            },
            "duplicate_rows": int(frame.duplicated().sum()),
            "numeric_summary": {},
            "categorical_summary": {},
            "correlation": {},
            "class_balance": {},
            "skewness": {},
            "outlier_iqr_counts": {},
        }

        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in frame.columns if c not in numeric_cols]

        if numeric_cols:
            report["numeric_summary"] = frame[numeric_cols].describe().round(4).to_dict()
            skew = frame[numeric_cols].skew(numeric_only=True).round(4)
            report["skewness"] = skew.to_dict()
            report["outlier_iqr_counts"] = {
                col: int(self._iqr_outlier_mask(frame[col]).sum()) for col in numeric_cols
            }
            if len(numeric_cols) >= 2:
                corr = frame[numeric_cols].corr().round(4)
                report["correlation"] = corr.to_dict()

        for col in categorical_cols:
            vc = frame[col].astype("string").value_counts(dropna=False).head(20)
            report["categorical_summary"][col] = {
                "n_unique": int(frame[col].nunique(dropna=False)),
                "top_values": {str(k): int(v) for k, v in vc.items()},
            }

        if target and target in frame.columns and context.task_type == TaskType.CLASSIFICATION:
            balance = frame[target].value_counts(normalize=True).round(4)
            report["class_balance"] = balance.to_dict()
            if balance.min() < 0.1:
                context.add_warning(
                    f"Severe class imbalance detected for '{target}' "
                    f"(min share={float(balance.min()):.2%})."
                )

        context.extras["eda_report"] = report

        out_dir = ensure_directory(context.config.save.output_dir)
        report_path = write_json(out_dir / "eda_report.json", report)
        context.store_artifact("eda_report", str(report_path))
        return context

    @staticmethod
    def _iqr_outlier_mask(series: pd.Series) -> pd.Series:
        """Boolean mask of IQR outliers for a numeric series."""
        clean = series.dropna()
        if clean.empty:
            return pd.Series(False, index=series.index)
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=series.index)
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return (series < lower) | (series > upper)
