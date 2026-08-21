"""Tune stage — Optuna Bayesian hyperparameter optimization.

Purpose:
    Refine the winning model family with Optuna and compare against baseline CV.

Interactions:
    - Reads ``best_model_name`` / preprocessor from context.
    - Replaces ``context.model`` when tuning beats the baseline.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from ml_pilot.config.schema import TaskType
from ml_pilot.core.context import PipelineContext
from ml_pilot.core.stage import BaseStage
from ml_pilot.stages.preprocess import transform_features


optuna.logging.set_verbosity(optuna.logging.WARNING)


class TuneStage(BaseStage):
    """Optuna-based hyperparameter search for the leading model family."""

    @property
    def name(self) -> str:
        return "tune"

    def should_skip(self, context: PipelineContext) -> bool:
        return not context.config.tune.enable or context.config.runtime.compare_only

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run Bayesian optimization and keep the better of tuned vs baseline.

        Args:
            context: Shared pipeline state.

        Returns:
            Updated context.
        """
        X_train, _, y_train, _ = context.require_splits()
        if context.preprocessor is None or not context.best_model_name:
            raise RuntimeError("Compare stage must run before tune.")

        cfg = context.config.tune
        seed = context.config.runtime.seed
        X_tr = transform_features(context.preprocessor, X_train)
        scoring = context.extras.get("compare_scoring") or (
            "f1_weighted" if context.task_type == TaskType.CLASSIFICATION else "r2"
        )

        factory = self._search_space(context.best_model_name, context.task_type, seed)
        if factory is None:
            context.add_warning(
                f"No search space for '{context.best_model_name}'; skipping tune."
            )
            return context

        baseline_model = context.extras.get("fitted_estimators", {}).get(context.best_model_name)
        if baseline_model is None:
            baseline_score = float("-inf")
        else:
            baseline_score = float(
                np.mean(
                    cross_val_score(
                        baseline_model,
                        X_tr,
                        y_train,
                        cv=cfg.cv_folds,
                        scoring=scoring,
                        n_jobs=context.config.compare.n_jobs,
                    )
                )
            )

        def objective(trial: optuna.Trial) -> float:
            estimator = factory(trial)
            scores = cross_val_score(
                estimator,
                X_tr,
                y_train,
                cv=cfg.cv_folds,
                scoring=scoring,
                n_jobs=context.config.compare.n_jobs,
            )
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=seed),
            study_name=f"mlpilot_{context.best_model_name}",
        )
        study.optimize(
            objective,
            n_trials=cfg.n_trials,
            timeout=cfg.timeout_seconds,
            show_progress_bar=False,
        )

        best_params = study.best_params
        tuned = factory(optuna.trial.FixedTrial(best_params))
        tuned_score = float(study.best_value)

        context.extras["tune_baseline_score"] = baseline_score
        context.extras["tune_best_score"] = tuned_score
        context.extras["tune_best_params"] = best_params

        chosen = tuned if tuned_score >= baseline_score else clone(baseline_model)
        label = "tuned" if tuned_score >= baseline_score else "baseline"
        context.extras["tune_winner"] = label

        pipe = Pipeline(
            steps=[
                ("preprocess", context.preprocessor),
                ("model", chosen),
            ]
        )
        pipe.fit(X_train, y_train)
        context.model = pipe
        context.best_model_name = f"{context.best_model_name}[{label}]"
        return context

    def _search_space(
        self,
        model_name: str,
        task: TaskType,
        seed: int,
    ) -> Callable[[optuna.Trial], Any] | None:
        """Return a trial→estimator factory for supported model families."""
        root = model_name.split("[", maxsplit=1)[0]

        if task == TaskType.CLASSIFICATION:
            return self._clf_factory(root, seed)
        return self._reg_factory(root, seed)

    @staticmethod
    def _clf_factory(root: str, seed: int) -> Callable[[optuna.Trial], Any] | None:
        if root == "random_forest":
            def factory(trial: optuna.Trial) -> Any:
                return RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 400),
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=seed,
                    n_jobs=-1,
                )
            return factory
        if root == "hist_gradient_boosting":
            def factory(trial: optuna.Trial) -> Any:
                return HistGradientBoostingClassifier(
                    max_depth=trial.suggest_int("max_depth", 3, 15),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_iter=trial.suggest_int("max_iter", 100, 400),
                    random_state=seed,
                )
            return factory
        if root == "gradient_boosting":
            def factory(trial: optuna.Trial) -> Any:
                return GradientBoostingClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int("max_depth", 2, 6),
                    random_state=seed,
                )
            return factory
        if root in {"logistic_regression", "ridge_classifier"}:
            def factory(trial: optuna.Trial) -> Any:
                return LogisticRegression(
                    C=trial.suggest_float("C", 1e-3, 10.0, log=True),
                    max_iter=2000,
                    random_state=seed,
                )
            return factory
        return None

    @staticmethod
    def _reg_factory(root: str, seed: int) -> Callable[[optuna.Trial], Any] | None:
        if root == "random_forest":
            def factory(trial: optuna.Trial) -> Any:
                return RandomForestRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 100, 400),
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=seed,
                    n_jobs=-1,
                )
            return factory
        if root == "hist_gradient_boosting":
            def factory(trial: optuna.Trial) -> Any:
                return HistGradientBoostingRegressor(
                    max_depth=trial.suggest_int("max_depth", 3, 15),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_iter=trial.suggest_int("max_iter", 100, 400),
                    random_state=seed,
                )
            return factory
        if root == "gradient_boosting":
            def factory(trial: optuna.Trial) -> Any:
                return GradientBoostingRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int("max_depth", 2, 6),
                    random_state=seed,
                )
            return factory
        if root == "ridge":
            def factory(trial: optuna.Trial) -> Any:
                return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 10.0, log=True))
            return factory
        if root == "elastic_net":
            def factory(trial: optuna.Trial) -> Any:
                return ElasticNet(
                    alpha=trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.05, 0.95),
                    max_iter=5000,
                    random_state=seed,
                )
            return factory
        return None
