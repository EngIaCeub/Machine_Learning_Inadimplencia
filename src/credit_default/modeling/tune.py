"""Hyperparameter tuning boundary.

Never use the final test set to choose hyperparameters or thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import ParameterSampler

from credit_default.config import get_project_config
from credit_default.modeling.evaluate import EvaluationResult


@dataclass(frozen=True)
class TuningResult:
    model_name: str
    best_estimator: Any
    best_params: dict[str, Any]
    best_cv_score: float
    refit_metric: str


COMPACT_SEARCH_SPACES = {
    "logistic_regression": {
        "C": [0.3, 1.0, 3.0],
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced", {0: 1.0, 1: 2.0}],
    },
    "decision_tree": {
        "max_depth": [6, 10, 14],
        "min_samples_leaf": [25, 50],
        "min_samples_split": [20, 80],
        "class_weight": [None, "balanced", {0: 1.0, 1: 2.0}],
    },
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [8, 12],
        "min_samples_leaf": [5, 25],
        "min_samples_split": [20, 50],
        "max_features": ["sqrt"],
        "class_weight": ["balanced_subsample", None, {0: 1.0, 1: 2.0}],
    },
    "hist_gradient_boosting": {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_iter": [200, 400],
        "max_leaf_nodes": [15, 31],
        "max_depth": [None, 6],
        "min_samples_leaf": [20, 50],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "xgboost": {
        "n_estimators": [200, 400, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [1, 5, 10],
        "reg_alpha": [0.0, 0.5, 1.0],
        "scale_pos_weight": [1.0, "light", "balanced"],
    },
}


def _resolve_xgboost_scale_pos_weight(value: float | str, target: pd.Series) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    positive_count = int((target == 1).sum())
    negative_count = int((target == 0).sum())
    balanced = negative_count / positive_count
    if value == "balanced":
        return float(balanced)
    if value == "light":
        return float((1.0 + balanced) / 2.0)
    raise ValueError(f"Unknown scale_pos_weight strategy: {value}.")


def select_promising_candidates(
    validation_results: list[EvaluationResult],
    max_candidates: int = 3,
) -> tuple[str, ...]:
    """Select a compact tuning set from the strongest non-baseline candidates."""

    eligible = [result for result in validation_results if result.model_name != "dummy_classifier"]
    eligible.sort(
        key=lambda result: (
            result.metrics["roc_auc"],
            result.metrics["f1"],
            result.metrics["recall"],
            result.metrics["precision"],
        ),
        reverse=True,
    )
    return tuple(result.model_name for result in eligible[:max_candidates])


def tune_model(
    model_name: str,
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series | None = None,
) -> TuningResult:
    """Run a compact deterministic search on train only."""

    if model_name not in COMPACT_SEARCH_SPACES:
        raise ValueError(f"No tuning space configured for {model_name}.")

    random_seed = get_project_config().random_seed
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    if model_name == "xgboost":
        search_space = COMPACT_SEARCH_SPACES[model_name]
        sampled_params = list(ParameterSampler(search_space, n_iter=18, random_state=random_seed))
        resolved_params = []
        for params in sampled_params:
            resolved = dict(params)
            resolved["scale_pos_weight"] = _resolve_xgboost_scale_pos_weight(
                resolved["scale_pos_weight"],
                target,
            )
            resolved_params.append({name: [value] for name, value in resolved.items()})
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=resolved_params,
            scoring={
                "roc_auc": "roc_auc",
                "f1": "f1",
                "recall": "recall",
                "average_precision": "average_precision",
            },
            refit="average_precision",
            cv=cv,
            n_jobs=1,
        )
        search.fit(features, target, **fit_kwargs)
        return TuningResult(
            model_name=model_name,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_cv_score=float(search.best_score_),
            refit_metric="average_precision",
        )

    search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=COMPACT_SEARCH_SPACES[model_name],
        scoring={
            "roc_auc": "roc_auc",
            "f1": "f1",
            "recall": "recall",
            "average_precision": "average_precision",
        },
        refit="average_precision",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(features, target, **fit_kwargs)
    return TuningResult(
        model_name=model_name,
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_cv_score=float(search.best_score_),
        refit_metric="average_precision",
    )
