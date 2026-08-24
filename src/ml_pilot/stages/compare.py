"""Compare stage — cross-validated multi-model leaderboard.

Purpose:
    Train 14+ sklearn estimators with CV and rank them by a primary metric.

Interactions:
    - Uses fitted ``context.preprocessor`` and train splits.
    - Writes ``comparison_table``, ``best_model_name``, and a baseline model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_pilot.config.schema import TaskType
from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage
from ml_pilot.stages.preprocess import transform_features
from ml_pilot.utils.io import ensure_directory, write_json


def _classification_zoo(seed: int) -> dict[str, Any]:
    """Return named classification estimators (14+ models)."""
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "sgd_classifier": SGDClassifier(loss="log_loss", random_state=seed, max_iter=2000),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
        "ada_boost": AdaBoostClassifier(random_state=seed),
        "bagging": BaggingClassifier(random_state=seed, n_jobs=-1),
        "knn": KNeighborsClassifier(),
        "gaussian_nb": GaussianNB(),
        "linear_svc": LinearSVC(random_state=seed, max_iter=5000),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed),
        "passive_aggressive": SGDClassifier(
            loss="hinge", penalty="l2", random_state=seed, max_iter=2000
        ),
    }


def _regression_zoo(seed: int) -> dict[str, Any]:
    """Return named regression estimators (14+ models)."""
    return {
        "ridge": Ridge(random_state=seed),
        "elastic_net": ElasticNet(random_state=seed, max_iter=5000),
        "sgd_regressor": SGDRegressor(random_state=seed, max_iter=2000),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
        "ada_boost": AdaBoostRegressor(random_state=seed),
        "bagging": BaggingRegressor(random_state=seed, n_jobs=-1),
        "knn": KNeighborsRegressor(),
        "linear_svr": LinearSVR(random_state=seed, max_iter=5000),
        "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed),
        "huber": SGDRegressor(loss="huber", random_state=seed, max_iter=2000),
    }

class CompareStage(BaseStage):
    """Head-to-head model comparison with cross-validation."""

    @property
    def name(self) -> str:
        return "compare"

    def run(self, context: PipelineContext) -> PipelineContext:
        """Evaluate candidate models and pick a provisional winner.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        X_train, _, y_train, _ = context.require_splits()
        if context.preprocessor is None:
            raise RuntimeError("Preprocessor missing; run preprocess stage first.")

        cfg = context.config.compare
        seed = context.config.runtime.seed
        X_tr = transform_features(context.preprocessor, X_train)

        if context.task_type == TaskType.CLASSIFICATION:
            zoo = _classification_zoo(seed)
            scoring = cfg.scoring or "f1_weighted"
            higher_is_better = True
        else:
            zoo = _regression_zoo(seed)
            scoring = cfg.scoring or "r2"
            higher_is_better = True

        if cfg.include_models:
            zoo = {k: v for k, v in zoo.items() if k in cfg.include_models}
        if cfg.exclude_models:
            zoo = {k: v for k, v in zoo.items() if k not in cfg.exclude_models}
        if not zoo:
            raise ValueError("No models left to compare after include/exclude filters.")

        rows: list[dict[str, Any]] = []
        fitted: dict[str, Any] = {}
        n_jobs = cfg.n_jobs
        failures: list[str] = []

        for name, estimator in zoo.items():
            try:
                cv_folds = min(cfg.cv_folds, len(y_train))
                
                if cv_folds < 2:
                    raise ValueError(
                        f"Need at least 2 training samples for cross-validation. Got {len(y_train)}."
                    )
                
                cv = CompareStage._safe_cross_validate(
                    estimator,
                    X_tr,
                    y_train,
                    cv_folds=cv_folds,
                    scoring=scoring,
                    n_jobs=n_jobs,
                )
                if cv.pop("_forced_serial", False):
                    n_jobs = 1
                model = clone(estimator)
                model.fit(X_tr, y_train)
                fitted[name] = model
                rows.append(
                    {
                        "model": name,
                        "cv_mean": float(np.mean(cv["test_score"])),
                        "cv_std": float(np.std(cv["test_score"])),
                        "train_mean": float(np.mean(cv["train_score"])),
                        "fit_time_mean": float(np.mean(cv["fit_time"])),
                    }
                )
            except Exception as exc:  
                msg = f"Model '{name}' failed during compare: {exc}"
                failures.append(msg)
                context.add_warning(msg)

        if not rows:
            detail = "; ".join(failures[:3]) if failures else "unknown error"
            raise RuntimeError(f"All candidate models failed during comparison. {detail}")

        table = pd.DataFrame(rows).sort_values(
            "cv_mean", ascending=not higher_is_better
        ).reset_index(drop=True)
        best_name = str(table.iloc[0]["model"])
        best_model = fitted[best_name]

        # Full sklearn Pipeline: preprocessor + estimator for later stages.
        full_pipe = Pipeline(
            steps=[
                ("preprocess", context.preprocessor),
                ("model", clone(best_model)),
            ]
        )
        full_pipe.fit(X_train, y_train)

        context.comparison_table = table
        context.best_model_name = best_name
        context.model = full_pipe
        context.extras["fitted_estimators"] = fitted
        context.extras["compare_scoring"] = scoring
        context.extras["transformed_feature_names"] = list(X_tr.columns)

        out_dir = ensure_directory(context.config.save.output_dir)
        path = write_json(out_dir / "model_comparison.json", table.to_dict(orient="records"))
        context.store_artifact("model_comparison", str(path))
        return context

    @staticmethod
    def _safe_cross_validate(
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        cv_folds: int,
        scoring: str,
        n_jobs: int,
    ) -> dict[str, Any]:
        """Run CV, falling back to ``n_jobs=1`` when the parallel backend fails."""
        try:
            result = cross_validate(
                estimator,
                X,
                y,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                return_train_score=True,
                error_score="raise",
            )
            result["_forced_serial"] = False
            return result
        except Exception:
            if n_jobs == 1:
                raise
            result = cross_validate(
                estimator,
                X,
                y,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=1,
                return_train_score=True,
                error_score="raise",
            )
            result["_forced_serial"] = True
            return result
