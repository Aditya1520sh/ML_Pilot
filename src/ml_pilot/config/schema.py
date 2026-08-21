"""Pydantic configuration schema for MLPilot.

Purpose:
    Define typed, validated settings for every pipeline stage and runtime flag.

Interactions:
    - Instantiated by ``config.resolver.load_config``.
    - Attached to ``PipelineContext.config`` and read by each stage.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    """Supported learning task kinds."""

    AUTO = "auto"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class DataConfig(BaseModel):
    """Input dataset settings."""

    path: str | None = None
    target: str | None = None
    task: TaskType = TaskType.AUTO
    id_columns: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)


class SplitConfig(BaseModel):
    """Train/test split settings."""

    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 42
    stratify: bool = True


class CleanConfig(BaseModel):
    """Data cleaning behaviour."""

    drop_duplicates: bool = True
    drop_constant: bool = True
    iqr_factor: float = Field(default=1.5, gt=0.0)
    remove_outliers: bool = False
    max_cardinality_ratio: float = Field(default=0.95, gt=0.0, le=1.0)
    min_non_null_ratio: float = Field(default=0.05, ge=0.0, le=1.0)


class FeatureEngConfig(BaseModel):
    """Feature engineering toggles."""

    datetime_decompose: bool = True
    log_skew_threshold: float = 1.0
    add_missingness_flags: bool = True
    pairwise_interactions: bool = False
    rare_category_threshold: float = Field(default=0.01, ge=0.0, le=1.0)


class PreprocessConfig(BaseModel):
    """ColumnTransformer preprocessing choices."""

    numeric_impute: Literal["mean", "median", "most_frequent"] = "median"
    categorical_impute: Literal["most_frequent", "constant"] = "most_frequent"
    scaler: Literal["standard", "minmax", "robust", "none"] = "standard"
    encoder: Literal["onehot", "ordinal"] = "onehot"


class CompareConfig(BaseModel):
    """Multi-model comparison settings."""

    cv_folds: int = Field(default=5, ge=2)
    n_jobs: int = -1
    scoring: str | None = None
    include_models: list[str] | None = None
    exclude_models: list[str] = Field(default_factory=list)


class TuneConfig(BaseModel):
    """Optuna hyperparameter search settings."""

    n_trials: int = Field(default=40, ge=1)
    timeout_seconds: int | None = None
    cv_folds: int = Field(default=3, ge=2)
    enable: bool = True


class SelectConfig(BaseModel):
    """Feature selection settings."""

    enable: bool = True
    top_k: int | None = Field(default=None, ge=1)
    importance_method: Literal["shap", "permutation"] = "shap"
    min_importance_share: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluateConfig(BaseModel):
    """Hold-out evaluation settings."""

    overfit_gap_threshold: float = Field(default=0.1, ge=0.0)


class ExplainConfig(BaseModel):
    """SHAP explainability settings."""

    enable: bool = True
    max_samples: int = Field(default=200, ge=10)
    plots: list[str] = Field(default_factory=lambda: ["summary", "dependence", "waterfall"])


class SaveConfig(BaseModel):
    """Artifact export settings."""

    output_dir: str = "mlpilot_artifacts"
    model_filename: str = "mlpilot_pipeline.joblib"
    metadata_filename: str = "run_metadata.json"


class RuntimeConfig(BaseModel):
    """Cross-cutting runtime flags."""

    verbose: bool = True
    compare_only: bool = False
    no_shap: bool = False
    seed: int = 42


class MLPilotConfig(BaseModel):
    """Root configuration for a full MLPilot run."""

    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    clean: CleanConfig = Field(default_factory=CleanConfig)
    feature_eng: FeatureEngConfig = Field(default_factory=FeatureEngConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    compare: CompareConfig = Field(default_factory=CompareConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)
    select: SelectConfig = Field(default_factory=SelectConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    save: SaveConfig = Field(default_factory=SaveConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("data", mode="before")
    @classmethod
    def _coerce_data(cls, value: Any) -> Any:
        return value or {}

    def model_dump_plain(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the config."""
        return self.model_dump(mode="json")
