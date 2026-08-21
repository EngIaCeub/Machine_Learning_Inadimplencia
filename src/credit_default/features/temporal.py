"""Explicit temporal trajectory features for the six monthly observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_default.features.credit_default_features import (
    BILL_AMOUNT_COLUMNS,
    PAYMENT_AMOUNT_COLUMNS,
    PAY_DELAY_COLUMNS,
    SEMANTIC_COLUMN_MAP,
    validate_credit_default_schema,
)

TEMPORAL_FEATURE_GROUPS = {
    "pay_trajectory": (
        "delay_worsening_count", "delay_improving_count", "delay_stable_count", "delay_transition_count",
        "delay_recent_vs_old", "delay_acceleration", "delay_recovery_count", "months_since_last_delay",
        "months_since_last_severe_delay", "longest_delay_streak", "longest_no_delay_streak",
        "recent_weighted_delay", "delay_direction", "delay_slope", "delay_curvature", "delay_recent_acceleration",
    ),
    "bill_payment_trajectory": (
        "bill_delta_1", "bill_delta_2", "bill_delta_3", "bill_delta_4", "bill_delta_5", "bill_pct_change",
        "bill_slope_temporal", "bill_acceleration", "bill_recent_vs_old", "bill_recency_weighted_mean",
        "bill_direction_changes", "bill_monotonic_increase_count", "bill_monotonic_decrease_count",
        "payment_delta_1", "payment_delta_2", "payment_delta_3", "payment_delta_4", "payment_delta_5",
        "payment_pct_change", "payment_slope_temporal", "payment_acceleration", "payment_recent_vs_old",
        "payment_recency_weighted_mean", "payment_increase_count", "payment_decrease_count", "payment_recovery_count",
    ),
    "shortfall_coverage": (
        "shortfall_mean", "shortfall_recent", "shortfall_max", "shortfall_slope", "shortfall_acceleration",
        "shortfall_positive_count", "shortfall_streak", "recent_shortfall_ratio", "coverage_mean", "coverage_recent",
        "coverage_min", "coverage_slope", "coverage_acceleration", "low_coverage_count", "low_coverage_streak",
        "coverage_recovery_count",
    ),
    "utilization_interactions": (
        "utilization_recent_temporal", "utilization_mean_temporal", "utilization_slope_temporal",
        "utilization_acceleration", "high_utilization_count", "utilization_increase_streak",
        "delay_worsening_utilization_increase", "delay_worsening_low_coverage", "recent_delay_shortfall",
        "delay_recovery_payment_recovery",
    ),
}

TEMPORAL_FEATURE_COLUMNS = tuple(column for group in TEMPORAL_FEATURE_GROUPS.values() for column in group)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe = denominator.where(denominator.abs() > 1e-12)
    return numerator.divide(safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _slope(values: pd.DataFrame) -> pd.Series:
    x = np.arange(values.shape[1], dtype=float)
    centered = x - x.mean()
    return pd.Series((values.to_numpy(dtype=float) * centered).sum(axis=1) / max(float((centered**2).sum()), 1.0), index=values.index)


def _streak(frame: pd.DataFrame, condition: pd.DataFrame) -> pd.Series:
    current = np.zeros(len(frame), dtype=int)
    longest = np.zeros(len(frame), dtype=int)
    for column in condition.columns:
        current = (current + condition[column].to_numpy(dtype=int)) * condition[column].to_numpy(dtype=int)
        longest = np.maximum(longest, current)
    return pd.Series(longest, index=frame.index, dtype=float)


def _add_pay(features: pd.DataFrame, pay: pd.DataFrame) -> None:
    delta = pay.diff(axis=1).iloc[:, 1:]
    change = np.sign(delta)
    features["delay_worsening_count"] = (delta > 0).sum(axis=1)
    features["delay_improving_count"] = (delta < 0).sum(axis=1)
    features["delay_stable_count"] = (delta == 0).sum(axis=1)
    features["delay_transition_count"] = (change.diff(axis=1).iloc[:, 1:] != 0).sum(axis=1)
    features["delay_recent_vs_old"] = pay.iloc[:, :3].mean(axis=1) - pay.iloc[:, 3:].mean(axis=1)
    features["delay_acceleration"] = delta.diff(axis=1).iloc[:, 1:].mean(axis=1)
    features["delay_recovery_count"] = (delta < 0).sum(axis=1)
    features["months_since_last_delay"] = pay.apply(lambda row: next((i for i, value in enumerate(row, 0) if value <= 0), len(row)), axis=1)
    features["months_since_last_severe_delay"] = pay.apply(lambda row: next((i for i, value in enumerate(row, 0) if value < 3), len(row)), axis=1)
    features["longest_delay_streak"] = _streak(pay, pay.gt(0))
    features["longest_no_delay_streak"] = _streak(pay, pay.le(0))
    features["recent_weighted_delay"] = np.average(pay.to_numpy(dtype=float), axis=1, weights=np.arange(1, 7)[::-1])
    features["delay_direction"] = np.sign(pay.iloc[:, 0] - pay.iloc[:, -1])
    features["delay_slope"] = _slope(pay)
    features["delay_curvature"] = delta.diff(axis=1).iloc[:, 1:].mean(axis=1)
    features["delay_recent_acceleration"] = delta.iloc[:, :2].mean(axis=1) - delta.iloc[:, -2:].mean(axis=1)


def build_temporal_trajectory_features(features: pd.DataFrame, enabled_groups: tuple[str, ...] | list[str] | None = None) -> pd.DataFrame:
    """Build target-free trajectory features while preserving row order."""
    validate_credit_default_schema(features.columns)
    groups = tuple(enabled_groups) if enabled_groups is not None else tuple(TEMPORAL_FEATURE_GROUPS)
    result = features.copy()
    pay = result.loc[:, PAY_DELAY_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    bill = result.loc[:, BILL_AMOUNT_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    payment = result.loc[:, PAYMENT_AMOUNT_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    limit_bal = pd.to_numeric(result["X1"], errors="coerce").fillna(0.0).abs()

    if "pay_trajectory" in groups:
        _add_pay(result, pay)
    if "bill_payment_trajectory" in groups:
        for prefix, frame in (("bill", bill), ("payment", payment)):
            deltas = frame.diff(axis=1)
            for index in range(1, 6):
                result[f"{prefix}_delta_{index}"] = deltas.iloc[:, index]
            result[f"{prefix}_pct_change"] = _safe_ratio(frame.iloc[:, 0] - frame.iloc[:, -1], frame.iloc[:, -1].abs())
            result[f"{prefix}_slope_temporal"] = _slope(frame)
            result[f"{prefix}_acceleration"] = deltas.diff(axis=1).iloc[:, 1:].mean(axis=1)
            result[f"{prefix}_recent_vs_old"] = frame.iloc[:, :3].mean(axis=1) - frame.iloc[:, 3:].mean(axis=1)
            result[f"{prefix}_recency_weighted_mean"] = np.average(frame.to_numpy(dtype=float), axis=1, weights=np.arange(1, 7)[::-1])
            result[f"{prefix}_increase_count"] = (deltas > 0).sum(axis=1)
            result[f"{prefix}_decrease_count"] = (deltas < 0).sum(axis=1)
            if prefix == "bill":
                result["bill_direction_changes"] = (np.sign(deltas).diff(axis=1).iloc[:, 1:] != 0).sum(axis=1)
                result["bill_monotonic_increase_count"] = (deltas > 0).sum(axis=1)
                result["bill_monotonic_decrease_count"] = (deltas < 0).sum(axis=1)
            else:
                result["payment_recovery_count"] = (deltas > 0).sum(axis=1)
    if "shortfall_coverage" in groups:
        shortfall = bill - payment
        coverage = pd.DataFrame({f"m{i}": _safe_ratio(payment.iloc[:, i], bill.iloc[:, i].abs()) for i in range(6)}, index=result.index)
        short_delta = shortfall.diff(axis=1)
        result["shortfall_mean"] = shortfall.mean(axis=1)
        result["shortfall_recent"] = shortfall.iloc[:, 0]
        result["shortfall_max"] = shortfall.max(axis=1)
        result["shortfall_slope"] = _slope(shortfall)
        result["shortfall_acceleration"] = short_delta.diff(axis=1).iloc[:, 1:].mean(axis=1)
        result["shortfall_positive_count"] = shortfall.gt(0).sum(axis=1)
        result["shortfall_streak"] = _streak(shortfall, shortfall.gt(0))
        result["recent_shortfall_ratio"] = _safe_ratio(shortfall.iloc[:, 0], bill.iloc[:, 0].abs())
        result["coverage_mean"] = coverage.mean(axis=1)
        result["coverage_recent"] = coverage.iloc[:, 0]
        result["coverage_min"] = coverage.min(axis=1)
        result["coverage_slope"] = _slope(coverage)
        result["coverage_acceleration"] = coverage.diff(axis=1).diff(axis=1).iloc[:, 2:].mean(axis=1)
        result["low_coverage_count"] = coverage.lt(0.5).sum(axis=1)
        result["low_coverage_streak"] = _streak(coverage, coverage.lt(0.5))
        result["coverage_recovery_count"] = (coverage.diff(axis=1).iloc[:, 1:] > 0).sum(axis=1)
    if "utilization_interactions" in groups:
        utilization = pd.DataFrame({f"m{i}": _safe_ratio(bill.iloc[:, i], limit_bal) for i in range(6)}, index=result.index)
        result["utilization_recent_temporal"] = utilization.iloc[:, 0]
        result["utilization_mean_temporal"] = utilization.mean(axis=1)
        result["utilization_slope_temporal"] = _slope(utilization)
        result["utilization_acceleration"] = utilization.diff(axis=1).diff(axis=1).iloc[:, 2:].mean(axis=1)
        result["high_utilization_count"] = utilization.gt(0.75).sum(axis=1)
        result["utilization_increase_streak"] = _streak(utilization, utilization.diff(axis=1).fillna(0).gt(0))
        result["delay_worsening_utilization_increase"] = result.get("delay_worsening_count", pay.gt(0).sum(axis=1)) * utilization.diff(axis=1).iloc[:, 1:].gt(0).sum(axis=1)
        result["delay_worsening_low_coverage"] = result.get("delay_worsening_count", pay.gt(0).sum(axis=1)) * (1.0 - utilization.iloc[:, 0].clip(upper=1.0))
        result["recent_delay_shortfall"] = pay.iloc[:, 0] * (bill.iloc[:, 0] - payment.iloc[:, 0])
        result["delay_recovery_payment_recovery"] = pay.diff(axis=1).iloc[:, 1:].lt(0).sum(axis=1) * payment.diff(axis=1).iloc[:, 1:].gt(0).sum(axis=1)

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
