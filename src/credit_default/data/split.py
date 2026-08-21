"""Deterministic stratified train/validation/test splits for S02."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default.config import ProjectConfig, get_project_config


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def _positive_rate(target: pd.Series) -> float:
    return float(target.mean())


def _validate_split_inputs(features: pd.DataFrame, target: pd.Series, config: ProjectConfig) -> None:
    config.validate()
    if len(features) != len(target):
        raise ValueError("Features and target must have matching row counts.")
    unique_values = set(target.unique().tolist())
    if unique_values != {0, 1}:
        raise ValueError(f"Target must be binary 0/1 before splitting. Got {sorted(unique_values)}.")


def make_splits(
    features: pd.DataFrame,
    target: pd.Series,
    config: ProjectConfig | None = None,
) -> DatasetSplits:
    """Create deterministic 70/15/15 stratified splits by default."""

    resolved_config = config or get_project_config()
    _validate_split_inputs(features, target, resolved_config)

    temp_size = resolved_config.validation_size + resolved_config.test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        target,
        train_size=resolved_config.train_size,
        test_size=temp_size,
        stratify=target,
        random_state=resolved_config.random_seed,
    )

    relative_test_size = resolved_config.test_size / temp_size
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=resolved_config.random_seed,
    )

    return DatasetSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def summarize_splits(splits: DatasetSplits) -> dict[str, dict[str, float | int]]:
    """Return gate-friendly shape and prevalence checks for each split."""

    summary: dict[str, dict[str, float | int]] = {}
    for split_name, features, target in (
        ("train", splits.X_train, splits.y_train),
        ("validation", splits.X_validation, splits.y_validation),
        ("test", splits.X_test, splits.y_test),
    ):
        summary[split_name] = {
            "rows": len(features),
            "features": features.shape[1],
            "positive_rate": _positive_rate(target),
        }
    return summary


def validate_distinct_splits(splits: DatasetSplits) -> None:
    """Guard against row overlap across train/validation/test."""

    train_index = set(splits.X_train.index.tolist())
    validation_index = set(splits.X_validation.index.tolist())
    test_index = set(splits.X_test.index.tolist())

    if train_index & validation_index:
        raise ValueError("Train and validation splits overlap.")
    if train_index & test_index:
        raise ValueError("Train and test splits overlap.")
    if validation_index & test_index:
        raise ValueError("Validation and test splits overlap.")
