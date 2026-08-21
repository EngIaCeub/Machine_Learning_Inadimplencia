"""Dataset acquisition for the UCI credit-default dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd

from credit_default.data.schema import validate_basic_shape

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as exc:  # pragma: no cover - exercised only in misconfigured environments.
    raise ImportError(
        "ucimlrepo is required for automated dataset acquisition. Install requirements.txt first."
    ) from exc


def _ensure_dataframe(features: Any) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.copy()
    raise TypeError("UCI features must resolve to a pandas DataFrame.")


def _ensure_single_target(target: Any) -> pd.Series:
    if isinstance(target, pd.Series):
        series = target.copy()
    elif isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("Target must contain exactly one column.")
        series = target.iloc[:, 0].copy()
    else:
        raise TypeError("UCI target must resolve to a pandas Series or single-column DataFrame.")
    return series


def normalize_binary_target(target: Any) -> pd.Series:
    """Resolve the UCI target into a 1D binary integer series."""

    series = _ensure_single_target(target)
    non_null_values = pd.Index(series.dropna().unique())
    if non_null_values.empty:
        raise ValueError("Target cannot be empty.")

    if series.isna().any():
        raise ValueError("Target contains missing values.")

    string_map = {"0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1}
    if series.dtype == bool:
        normalized = series.astype(int)
    elif pd.api.types.is_numeric_dtype(series):
        normalized = pd.to_numeric(series, errors="raise").astype(int)
    else:
        lowered = series.astype(str).str.strip().str.lower()
        if not set(lowered.unique()).issubset(string_map):
            raise ValueError("Target must be binary and explicitly mappable to 0/1.")
        normalized = lowered.map(string_map).astype(int)

    unique_values = set(normalized.unique().tolist())
    if unique_values != {0, 1}:
        raise ValueError(f"Target must be binary 0/1. Got {sorted(unique_values)}.")

    normalized.name = series.name or "target"
    return normalized


def load_uci_dataset(dataset_id: int = 350) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI dataset id=350 into reusable tabular objects."""

    dataset = fetch_ucirepo(id=dataset_id)
    features = _ensure_dataframe(dataset.data.features)
    target = normalize_binary_target(dataset.data.targets)

    validate_basic_shape(n_rows=len(features), n_features=features.shape[1])
    if len(features) != len(target):
        raise ValueError("Features and target must have the same number of rows.")

    return features, target
