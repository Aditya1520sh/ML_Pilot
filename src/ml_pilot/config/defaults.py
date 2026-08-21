"""Default configuration values for MLPilot runs.

Purpose:
    Central source of built-in defaults used when no YAML config is supplied
    and when merging partial user overrides.

Interactions:
    - Consumed by ``config.resolver.load_config`` and ``config.schema.MLPilotConfig``.
    - Stages read resolved values via ``PipelineContext.config``.
"""

from __future__ import annotations

from typing import Any

DEFAULT_CONFIG_DICT: dict[str, Any] = {
    "data": {
        "path": None,
        "target": None,
        "task": "auto",
        "id_columns": [],
        "drop_columns": [],
    },
    "split": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    },
    "clean": {
        "drop_duplicates": True,
        "drop_constant": True,
        "iqr_factor": 1.5,
        "remove_outliers": False,
        "max_cardinality_ratio": 0.95,
        "min_non_null_ratio": 0.05,
    },
    "feature_eng": {
        "datetime_decompose": True,
        "log_skew_threshold": 1.0,
        "add_missingness_flags": True,
        "pairwise_interactions": False,
        "rare_category_threshold": 0.01,
    },
    "preprocess": {
        "numeric_impute": "median",
        "categorical_impute": "most_frequent",
        "scaler": "standard",
        "encoder": "onehot",
    },
    "compare": {
        "cv_folds": 5,
        "n_jobs": 1,
        "scoring": None,
        "include_models": None,
        "exclude_models": [],
    },
    "tune": {
        "n_trials": 40,
        "timeout_seconds": None,
        "cv_folds": 3,
        "enable": True,
    },
    "select": {
        "enable": True,
        "top_k": None,
        "importance_method": "shap",
        "min_importance_share": 0.0,
    },
    "evaluate": {
        "overfit_gap_threshold": 0.1,
    },
    "explain": {
        "enable": True,
        "max_samples": 200,
        "plots": ["summary", "dependence", "waterfall"],
    },
    "save": {
        "output_dir": "mlpilot_artifacts",
        "model_filename": "mlpilot_pipeline.joblib",
        "metadata_filename": "run_metadata.json",
    },
    "runtime": {
        "verbose": True,
        "compare_only": False,
        "no_shap": False,
        "seed": 42,
    },
}
