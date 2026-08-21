"""Schema validation and lightweight EDA helpers for S02."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

EXPECTED_MIN_ROWS = 29_000
EXPECTED_MIN_FEATURES = 20
DEFAULT_IDENTIFIER_COLUMNS = frozenset({"ID"})


@dataclass(frozen=True)
class SchemaReport:
    n_rows: int
    n_features: int
    target_name: str
    duplicate_rows: int
    missing_by_column: dict[str, int]
    dtype_by_column: dict[str, str]
    schema_issues: tuple[str, ...]
    leakage_risks: tuple[str, ...]
    target_positive_rate: float


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_basic_shape(n_rows: int, n_features: int) -> None:
    if n_rows < EXPECTED_MIN_ROWS:
        raise ValueError(f"Unexpectedly small dataset: {n_rows} rows.")
    if n_features < EXPECTED_MIN_FEATURES:
        raise ValueError(f"Unexpectedly small feature set: {n_features} features.")


def build_schema_report(
    features: pd.DataFrame,
    target: pd.Series,
    identifier_columns: set[str] | frozenset[str] = DEFAULT_IDENTIFIER_COLUMNS,
) -> SchemaReport:
    """Collect shape, type, missing, duplicate, and leakage checks."""

    validate_basic_shape(n_rows=len(features), n_features=features.shape[1])
    if len(features) != len(target):
        raise ValueError("Features and target must have matching row counts.")

    issues: list[str] = []
    if features.columns.duplicated().any():
        issues.append("duplicate feature names")
    if target.name in set(features.columns):
        issues.append("target name collides with feature name")

    duplicate_rows = int(features.duplicated().sum())
    if duplicate_rows:
        issues.append("duplicate rows")

    missing_by_column = {
        column: int(count)
        for column, count in features.isna().sum().items()
        if int(count) > 0
    }
    if missing_by_column:
        issues.append("missing values")

    unique_target_values = set(target.unique().tolist())
    if unique_target_values != {0, 1}:
        raise ValueError(f"Target must be normalized to binary 0/1. Got {sorted(unique_target_values)}.")

    present_identifiers = tuple(column for column in features.columns if column in identifier_columns)
    dtype_by_column = {column: str(dtype) for column, dtype in features.dtypes.items()}

    return SchemaReport(
        n_rows=len(features),
        n_features=features.shape[1],
        target_name=target.name or "target",
        duplicate_rows=duplicate_rows,
        missing_by_column=missing_by_column,
        dtype_by_column=dtype_by_column,
        schema_issues=tuple(issues),
        leakage_risks=present_identifiers,
        target_positive_rate=float(target.mean()),
    )


def build_eda_summary(
    features: pd.DataFrame,
    target: pd.Series,
    max_categories: int = 10,
) -> dict[str, Any]:
    """Return a serializable data-stage EDA summary for notebook/report orchestration."""

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_summary: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        describe = features[column].describe(percentiles=[0.25, 0.5, 0.75])
        numeric_summary[column] = {
            str(stat): float(_to_python_scalar(value))
            for stat, value in describe.items()
            if pd.notna(value)
        }

    categorical_summary: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        counts = features[column].astype("object").fillna("<MISSING>").value_counts(dropna=False)
        categorical_summary[column] = {
            str(category): int(count)
            for category, count in counts.head(max_categories).items()
        }

    target_counts = target.value_counts().sort_index()
    return {
        "shape": {"rows": len(features), "features": features.shape[1]},
        "target_balance": {
            "counts": {int(label): int(count) for label, count in target_counts.items()},
            "positive_rate": float(target.mean()),
        },
        "numeric_distributions": numeric_summary,
        "categorical_distributions": categorical_summary,
    }
