"""Clean stage — duplicates, constants, sparse columns, optional IQR trim.

Purpose:
    Produce a cleaner working frame before feature engineering.

Interactions:
    - Reads/writes ``context.frame``.
    - Uses ``config.clean`` settings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage


class CleanStage(BaseStage):
    """Deterministic tabular cleaning transforms."""

    @property
    def name(self) -> str:
        return "clean"

    def run(self, context: PipelineContext) -> PipelineContext:
        """Clean the working dataframe in place on the context.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        frame = context.require_frame().copy()
        target = context.target_name
        cfg = context.config.clean
        notes: list[str] = []

        if cfg.drop_duplicates:
            before = len(frame)
            frame = frame.drop_duplicates()
            removed = before - len(frame)
            if removed:
                notes.append(f"Removed {removed} duplicate rows.")

        # Drop near-empty columns (excluding target).
        non_null_ratio = frame.notna().mean()
        sparse = [
            col
            for col, ratio in non_null_ratio.items()
            if ratio < cfg.min_non_null_ratio and col != target
        ]
        if sparse:
            frame = frame.drop(columns=sparse)
            notes.append(f"Dropped sparse columns: {sparse}")

        if cfg.drop_constant:
            constants = [
                col
                for col in frame.columns
                if col != target and frame[col].nunique(dropna=False) <= 1
            ]
            if constants:
                frame = frame.drop(columns=constants)
                notes.append(f"Dropped constant columns: {constants}")

        # High-cardinality warnings / optional ID-like drops.
        n_rows = max(len(frame), 1)
        for col in list(frame.columns):
            if col == target:
                continue
            cardinality_ratio = frame[col].nunique(dropna=False) / n_rows
            if cardinality_ratio >= cfg.max_cardinality_ratio:
                context.add_warning(
                    f"Column '{col}' has very high cardinality "
                    f"({cardinality_ratio:.1%}); consider dropping as an ID."
                )
                # Auto-drop string columns that look like unique IDs.
                if frame[col].dtype == object or str(frame[col].dtype) == "string":
                    frame = frame.drop(columns=[col])
                    notes.append(f"Dropped likely ID column '{col}'.")

        if cfg.remove_outliers and target is not None:
            numeric_cols = [
                c
                for c in frame.select_dtypes(include=[np.number]).columns
                if c != target
            ]
            mask = pd.Series(True, index=frame.index)
            for col in numeric_cols:
                mask &= ~self._iqr_mask(frame[col], cfg.iqr_factor)
            before = len(frame)
            frame = frame.loc[mask].reset_index(drop=True)
            notes.append(f"IQR outlier removal dropped {before - len(frame)} rows.")

        # Remove rows where target is missing
        if target is not None and target in frame.columns:
            before = len(frame)
            frame = frame.dropna(subset=[target]).reset_index(drop=True)
            removed = before - len(frame)

            if removed:
                notes.append(
                f"Removed {removed} rows with missing target values."
                )


        context.frame = frame
        context.extras["clean_notes"] = notes
        return context

    @staticmethod
    def _iqr_mask(series: pd.Series, factor: float) -> pd.Series:
        """Return True for IQR outliers."""
        clean = series.dropna()
        if clean.empty:
            return pd.Series(False, index=series.index)
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=series.index)
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        return (series < lower) | (series > upper)
