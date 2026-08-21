"""Objective, reusable EDA helpers for S02."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_PRIORITY_COLUMNS = (
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "PAY_AMT1",
    "PAY_AMT2",
)
RELATION_PRIORITY_COLUMNS = ("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3")
CATEGORICAL_EXPECTATIONS = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}
REPAYMENT_STATUS_COLUMNS = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
VALID_REPAYMENT_STATUS = set(range(-2, 10))


@dataclass(frozen=True)
class EDAReport:
    target_distribution: pd.DataFrame
    descriptive_stats: pd.DataFrame
    missing_values: pd.DataFrame
    duplicate_summary: pd.DataFrame
    numeric_distribution_summary: pd.DataFrame
    default_rate_by_category: dict[str, pd.DataFrame]
    default_rate_by_numeric_bin: dict[str, pd.DataFrame]
    inconsistent_values: pd.DataFrame
    selected_numeric_columns: tuple[str, ...]
    selected_relation_columns: tuple[str, ...]


def _select_columns(
    frame: pd.DataFrame,
    priority_columns: tuple[str, ...],
    max_columns: int,
) -> tuple[str, ...]:
    selected = [column for column in priority_columns if column in frame.columns]
    if len(selected) < max_columns:
        for column in frame.columns:
            if column not in selected:
                selected.append(column)
            if len(selected) == max_columns:
                break
    return tuple(selected[:max_columns])


def build_target_distribution(target: pd.Series) -> pd.DataFrame:
    counts = target.value_counts().sort_index()
    return pd.DataFrame(
        {
            "target": counts.index.astype(int),
            "count": counts.values.astype(int),
            "rate": (counts / len(target)).values.astype(float),
        }
    )


def build_missing_values_table(features: pd.DataFrame) -> pd.DataFrame:
    missing_counts = features.isna().sum().sort_values(ascending=False)
    table = pd.DataFrame(
        {
            "column": missing_counts.index,
            "missing_count": missing_counts.values.astype(int),
            "missing_rate": (missing_counts / len(features)).values.astype(float),
        }
    )
    return table.loc[table["missing_count"] > 0].reset_index(drop=True)


def build_duplicate_summary(features: pd.DataFrame) -> pd.DataFrame:
    duplicate_rows = int(features.duplicated().sum())
    return pd.DataFrame([{"duplicate_rows": duplicate_rows, "duplicate_rate": duplicate_rows / len(features)}])


def build_descriptive_stats(features: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    stats = features.loc[:, list(columns)].describe(percentiles=[0.25, 0.5, 0.75]).transpose()
    stats.index.name = "feature"
    return stats.reset_index()


def build_numeric_distribution_summary(features: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in columns:
        series = pd.to_numeric(features[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
            }
        )
    return pd.DataFrame(rows)


def build_default_rate_by_category(
    features: pd.DataFrame,
    target: pd.Series,
    columns: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    combined = features.copy()
    combined["_target_"] = target.values
    results: dict[str, pd.DataFrame] = {}
    for column in columns:
        series = combined[column]
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=False) > 12:
            continue
        grouped = (
            combined.assign(**{column: series.fillna("<MISSING>")})
            .groupby(column, dropna=False)["_target_"]
            .agg(default_rate="mean", observations="size")
            .reset_index()
            .sort_values(["default_rate", "observations"], ascending=[False, False])
        )
        results[column] = grouped
    return results


def build_default_rate_by_numeric_bin(
    features: pd.DataFrame,
    target: pd.Series,
    columns: tuple[str, ...],
    bins: int = 5,
) -> dict[str, pd.DataFrame]:
    combined = features.copy()
    combined["_target_"] = target.values
    results: dict[str, pd.DataFrame] = {}
    for column in columns:
        series = pd.to_numeric(combined[column], errors="coerce")
        unique_values = int(series.nunique(dropna=True))
        effective_bins = min(bins, unique_values)
        if effective_bins < 2:
            continue
        binned = pd.qcut(series, q=effective_bins, duplicates="drop")
        grouped = (
            pd.DataFrame({"bin": binned.astype(str), "target": combined["_target_"]})
            .groupby("bin", dropna=False)["target"]
            .agg(default_rate="mean", observations="size")
            .reset_index()
        )
        results[column] = grouped
    return results


def build_inconsistent_values_table(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column, valid_values in CATEGORICAL_EXPECTATIONS.items():
        if column not in features.columns:
            continue
        observed = set(pd.Series(features[column]).dropna().astype(int).unique().tolist())
        unexpected = sorted(value for value in observed if value not in valid_values)
        if unexpected:
            rows.append(
                {
                    "column": column,
                    "issue": "unexpected_category",
                    "details": ",".join(str(value) for value in unexpected),
                }
            )

    for column in REPAYMENT_STATUS_COLUMNS:
        if column not in features.columns:
            continue
        observed = set(pd.Series(features[column]).dropna().astype(int).unique().tolist())
        unexpected = sorted(value for value in observed if value not in VALID_REPAYMENT_STATUS)
        if unexpected:
            rows.append(
                {
                    "column": column,
                    "issue": "unexpected_repayment_status",
                    "details": ",".join(str(value) for value in unexpected),
                }
            )

    if "AGE" in features.columns:
        age_series = pd.to_numeric(features["AGE"], errors="coerce")
        invalid_age_count = int(((age_series < 18) | (age_series > 100)).fillna(False).sum())
        if invalid_age_count:
            rows.append(
                {
                    "column": "AGE",
                    "issue": "possible_age_outlier",
                    "details": str(invalid_age_count),
                }
            )

    return pd.DataFrame(rows, columns=["column", "issue", "details"])


def build_eda_report(
    features: pd.DataFrame,
    target: pd.Series,
    max_numeric_columns: int = 6,
    max_relation_columns: int = 6,
) -> EDAReport:
    numeric_candidates = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    selected_numeric_columns = _select_columns(
        features.loc[:, numeric_candidates],
        NUMERIC_PRIORITY_COLUMNS,
        max_columns=min(max_numeric_columns, len(numeric_candidates)),
    )
    relation_candidates = [column for column in RELATION_PRIORITY_COLUMNS if column in features.columns]
    if len(relation_candidates) < max_relation_columns:
        for column in features.columns:
            if column not in relation_candidates and features[column].nunique(dropna=False) <= 12:
                relation_candidates.append(column)
            if len(relation_candidates) == max_relation_columns:
                break
    selected_relation_columns = tuple(relation_candidates[:max_relation_columns])

    return EDAReport(
        target_distribution=build_target_distribution(target),
        descriptive_stats=build_descriptive_stats(features, selected_numeric_columns),
        missing_values=build_missing_values_table(features),
        duplicate_summary=build_duplicate_summary(features),
        numeric_distribution_summary=build_numeric_distribution_summary(features, selected_numeric_columns),
        default_rate_by_category=build_default_rate_by_category(
            features,
            target,
            selected_relation_columns,
        ),
        default_rate_by_numeric_bin=build_default_rate_by_numeric_bin(
            features,
            target,
            selected_numeric_columns,
        ),
        inconsistent_values=build_inconsistent_values_table(features),
        selected_numeric_columns=selected_numeric_columns,
        selected_relation_columns=selected_relation_columns,
    )


def plot_target_distribution(target_distribution: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(target_distribution["target"].astype(str), target_distribution["count"], color="#1f77b4")
    axis.set_title("Target Distribution")
    axis.set_xlabel("Default")
    axis.set_ylabel("Count")
    figure.tight_layout()
    return figure


def plot_numeric_distributions(features: pd.DataFrame, columns: tuple[str, ...]) -> plt.Figure:
    subplot_columns = 2
    subplot_rows = ceil(len(columns) / subplot_columns)
    figure, axes = plt.subplots(subplot_rows, subplot_columns, figsize=(12, 4 * subplot_rows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, column in zip(axes_list, columns):
        axis.hist(pd.to_numeric(features[column], errors="coerce").dropna(), bins=20, color="#4c78a8")
        axis.set_title(column)
        axis.set_xlabel("Value")
        axis.set_ylabel("Count")

    for axis in axes_list[len(columns) :]:
        axis.axis("off")

    figure.tight_layout()
    return figure


def plot_default_rate_by_category(category_tables: dict[str, pd.DataFrame]) -> plt.Figure:
    subplot_columns = 2
    subplot_rows = ceil(max(len(category_tables), 1) / subplot_columns)
    figure, axes = plt.subplots(subplot_rows, subplot_columns, figsize=(12, 4 * subplot_rows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, (column, table) in zip(axes_list, category_tables.items()):
        labels = table[column].astype(str).tolist()
        axis.bar(labels, table["default_rate"], color="#f58518")
        axis.set_title(f"Default Rate by {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Default Rate")
        axis.tick_params(axis="x", rotation=45)

    for axis in axes_list[len(category_tables) :]:
        axis.axis("off")

    figure.tight_layout()
    return figure
