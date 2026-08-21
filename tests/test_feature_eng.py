"""Tests for TabularFeatureEngineer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_pilot.stages.feature_eng import TabularFeatureEngineer


def test_feature_engineer_adds_flags_and_log() -> None:
    frame = pd.DataFrame(
        {
            "skewed": [0.0, 1.0, 2.0, 50.0, 100.0, 200.0],
            "cat": ["a", "a", "a", "a", "a", "rare"],
            "with_nan": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        }
    )
    eng = TabularFeatureEngineer(
        log_skew_threshold=0.5,
        add_missingness_flags=True,
        rare_category_threshold=0.2,
    )
    eng.fit(frame)
    out = eng.transform(frame)
    assert "with_nan__was_missing" in out.columns
    assert "skewed__log1p" in out.columns
    assert (out["cat"] == "__rare__").sum() >= 1
