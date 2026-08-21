"""Semantic audit and feature engineering for the UCI credit-default dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEMANTIC_COLUMN_MAP = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}
RAW_CATEGORICAL_COLUMNS = ("X2", "X3", "X4")
PAY_DELAY_COLUMNS = ("X6", "X7", "X8", "X9", "X10", "X11")
BILL_AMOUNT_COLUMNS = ("X12", "X13", "X14", "X15", "X16", "X17")
PAYMENT_AMOUNT_COLUMNS = ("X18", "X19", "X20", "X21", "X22", "X23")

ROUND1_FEATURE_COLUMNS = (
    "count_pay_delay_gt0",
    "max_pay_delay",
    "recent_pay_delay",
    "mean_pay_delay",
    "pay_delay_trend",
    "avg_bill_amt",
    "max_bill_amt",
    "min_bill_amt",
    "bill_std",
    "bill_growth",
    "avg_pay_amt",
    "max_pay_amt",
    "pay_std",
    "pay_to_bill_ratio",
    "utilization_proxy",
)
PAY_BEHAVIOR_FEATURE_COLUMNS = (
    "pay_delay_count",
    "pay_delay_max",
    "pay_delay_mean",
    "pay_delay_std",
    "pay_recent_delay",
    "pay_recent_3m_mean",
    "pay_old_3m_mean",
    "pay_delay_trend_v2",
    "pay_consecutive_delay",
    "pay_any_severe_delay",
)
BILL_BEHAVIOR_FEATURE_COLUMNS = (
    "bill_mean",
    "bill_median",
    "bill_max",
    "bill_min",
    "bill_std_v2",
    "bill_range",
    "bill_recent",
    "bill_recent_3m_mean",
    "bill_previous_3m_mean",
    "bill_trend",
    "bill_change_recent",
    "bill_growth_ratio",
    "bill_cv",
)
PAYMENT_BEHAVIOR_FEATURE_COLUMNS = (
    "payment_mean",
    "payment_median",
    "payment_max",
    "payment_min",
    "payment_std_v2",
    "payment_total",
    "payment_recent",
    "payment_recent_3m_mean",
    "payment_trend",
    "payment_change_recent",
    "payment_cv",
)
RATIO_FEATURE_COLUMNS = (
    "payment_to_bill_ratio_m1",
    "payment_to_bill_ratio_m2",
    "payment_to_bill_ratio_m3",
    "payment_to_bill_ratio_m4",
    "payment_to_bill_ratio_m5",
    "payment_to_bill_ratio_m6",
    "payment_to_bill_ratio_mean",
    "payment_to_bill_ratio_min",
    "payment_to_bill_ratio_recent",
    "utilization_m1",
    "utilization_m2",
    "utilization_m3",
    "utilization_m4",
    "utilization_m5",
    "utilization_m6",
    "utilization_mean",
    "utilization_max",
    "utilization_recent",
    "utilization_trend",
    "utilization_cv",
)
TREND_FEATURE_COLUMNS = (
    "bill_slope",
    "payment_slope",
    "utilization_slope",
    "delay_slope",
    "payment_consistency",
    "bill_volatility",
    "payment_volatility",
    "delay_utilization_interaction",
    "delay_payment_ratio_interaction",
    "delay_count_utilization_interaction",
    "severe_delay_low_payment_interaction",
)
ROUND2_FEATURE_GROUPS = {
    "round1": ROUND1_FEATURE_COLUMNS,
    "pay": PAY_BEHAVIOR_FEATURE_COLUMNS,
    "bill": BILL_BEHAVIOR_FEATURE_COLUMNS,
    "payment": PAYMENT_BEHAVIOR_FEATURE_COLUMNS,
    "ratios": RATIO_FEATURE_COLUMNS,
    "trends": TREND_FEATURE_COLUMNS,
}
ENGINEERED_FEATURE_COLUMNS = (
    *ROUND1_FEATURE_COLUMNS,
    *PAY_BEHAVIOR_FEATURE_COLUMNS,
    *BILL_BEHAVIOR_FEATURE_COLUMNS,
    *PAYMENT_BEHAVIOR_FEATURE_COLUMNS,
    *RATIO_FEATURE_COLUMNS,
    *TREND_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class SemanticAuditReport:
    column_map: dict[str, str]
    categorical_columns: tuple[str, ...]
    ordinal_delay_columns: tuple[str, ...]
    bill_columns: tuple[str, ...]
    payment_columns: tuple[str, ...]


def validate_credit_default_schema(columns: pd.Index | list[str] | tuple[str, ...]) -> None:
    missing = [column for column in SEMANTIC_COLUMN_MAP if column not in columns]
    if missing:
        raise ValueError(f"Missing required credit-default columns: {missing}.")
    if "Y" in columns:
        raise ValueError("Target column must not be included in feature engineering input.")


def has_credit_default_schema(columns: pd.Index | list[str] | tuple[str, ...]) -> bool:
    return all(column in columns for column in SEMANTIC_COLUMN_MAP) and "Y" not in columns


def audit_credit_default_semantics(columns: pd.Index | list[str] | tuple[str, ...]) -> SemanticAuditReport:
    validate_credit_default_schema(columns)
    return SemanticAuditReport(
        column_map=SEMANTIC_COLUMN_MAP.copy(),
        categorical_columns=RAW_CATEGORICAL_COLUMNS,
        ordinal_delay_columns=PAY_DELAY_COLUMNS,
        bill_columns=BILL_AMOUNT_COLUMNS,
        payment_columns=PAYMENT_AMOUNT_COLUMNS,
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.where(denominator.abs() > 1e-12)
    ratio = numerator.divide(safe_denominator)
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rowwise_trend(frame: pd.DataFrame) -> pd.Series:
    positions = np.arange(frame.shape[1], dtype=float)
    centered = positions - positions.mean()
    denominator = float((centered**2).sum())
    values = frame.to_numpy(dtype=float)
    trend = (values * centered).sum(axis=1) / denominator
    return pd.Series(trend, index=frame.index, dtype=float)


def _rowwise_consecutive_positive(frame: pd.DataFrame) -> pd.Series:
    values = frame.gt(0).to_numpy(dtype=int)
    longest = np.zeros(values.shape[0], dtype=int)
    current = np.zeros(values.shape[0], dtype=int)
    for col in range(values.shape[1]):
        current = (current + values[:, col]) * values[:, col]
        longest = np.maximum(longest, current)
    return pd.Series(longest.astype(float), index=frame.index)


def _add_round1_features(engineered: pd.DataFrame, pay_delay: pd.DataFrame, bill_amounts: pd.DataFrame, payment_amounts: pd.DataFrame, limit_bal: pd.Series) -> None:
    engineered["count_pay_delay_gt0"] = pay_delay.gt(0).sum(axis=1).astype(float)
    engineered["max_pay_delay"] = pay_delay.max(axis=1)
    engineered["recent_pay_delay"] = pay_delay["X6"]
    engineered["mean_pay_delay"] = pay_delay.mean(axis=1)
    engineered["pay_delay_trend"] = _rowwise_trend(pay_delay)

    engineered["avg_bill_amt"] = bill_amounts.mean(axis=1)
    engineered["max_bill_amt"] = bill_amounts.max(axis=1)
    engineered["min_bill_amt"] = bill_amounts.min(axis=1)
    engineered["bill_std"] = bill_amounts.std(axis=1, ddof=0)
    engineered["bill_growth"] = bill_amounts["X12"] - bill_amounts["X17"]

    engineered["avg_pay_amt"] = payment_amounts.mean(axis=1)
    engineered["max_pay_amt"] = payment_amounts.max(axis=1)
    engineered["pay_std"] = payment_amounts.std(axis=1, ddof=0)

    engineered["pay_to_bill_ratio"] = _safe_ratio(engineered["avg_pay_amt"], engineered["avg_bill_amt"].abs())
    engineered["utilization_proxy"] = _safe_ratio(engineered["avg_bill_amt"], limit_bal.abs())


def build_behavioral_features(
    features: pd.DataFrame,
    *,
    enabled_groups: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Create deterministic behavioral features without using the target."""

    validate_credit_default_schema(features.columns)
    groups = tuple(enabled_groups) if enabled_groups is not None else tuple(ROUND2_FEATURE_GROUPS)
    engineered = features.copy()

    pay_delay = engineered.loc[:, PAY_DELAY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bill_amounts = engineered.loc[:, BILL_AMOUNT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    payment_amounts = engineered.loc[:, PAYMENT_AMOUNT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    limit_bal = pd.to_numeric(engineered["X1"], errors="coerce")

    if "round1" in groups:
        _add_round1_features(engineered, pay_delay, bill_amounts, payment_amounts, limit_bal)

    if "pay" in groups:
        engineered["pay_delay_count"] = pay_delay.gt(0).sum(axis=1).astype(float)
        engineered["pay_delay_max"] = pay_delay.max(axis=1)
        engineered["pay_delay_mean"] = pay_delay.mean(axis=1)
        engineered["pay_delay_std"] = pay_delay.std(axis=1, ddof=0)
        engineered["pay_recent_delay"] = pay_delay["X6"]
        engineered["pay_recent_3m_mean"] = pay_delay.loc[:, ("X6", "X7", "X8")].mean(axis=1)
        engineered["pay_old_3m_mean"] = pay_delay.loc[:, ("X9", "X10", "X11")].mean(axis=1)
        engineered["pay_delay_trend_v2"] = _rowwise_trend(pay_delay)
        engineered["pay_consecutive_delay"] = _rowwise_consecutive_positive(pay_delay)
        engineered["pay_any_severe_delay"] = pay_delay.ge(3).any(axis=1).astype(float)

    if "bill" in groups:
        engineered["bill_mean"] = bill_amounts.mean(axis=1)
        engineered["bill_median"] = bill_amounts.median(axis=1)
        engineered["bill_max"] = bill_amounts.max(axis=1)
        engineered["bill_min"] = bill_amounts.min(axis=1)
        engineered["bill_std_v2"] = bill_amounts.std(axis=1, ddof=0)
        engineered["bill_range"] = engineered["bill_max"] - engineered["bill_min"]
        engineered["bill_recent"] = bill_amounts["X12"]
        engineered["bill_recent_3m_mean"] = bill_amounts.loc[:, ("X12", "X13", "X14")].mean(axis=1)
        engineered["bill_previous_3m_mean"] = bill_amounts.loc[:, ("X15", "X16", "X17")].mean(axis=1)
        engineered["bill_trend"] = _rowwise_trend(bill_amounts)
        engineered["bill_change_recent"] = bill_amounts["X12"] - bill_amounts["X13"]
        engineered["bill_growth_ratio"] = _safe_ratio(bill_amounts["X12"] - bill_amounts["X17"], bill_amounts["X17"].abs())
        engineered["bill_cv"] = _safe_ratio(engineered["bill_std_v2"], engineered["bill_mean"].abs())

    if "payment" in groups:
        engineered["payment_mean"] = payment_amounts.mean(axis=1)
        engineered["payment_median"] = payment_amounts.median(axis=1)
        engineered["payment_max"] = payment_amounts.max(axis=1)
        engineered["payment_min"] = payment_amounts.min(axis=1)
        engineered["payment_std_v2"] = payment_amounts.std(axis=1, ddof=0)
        engineered["payment_total"] = payment_amounts.sum(axis=1)
        engineered["payment_recent"] = payment_amounts["X18"]
        engineered["payment_recent_3m_mean"] = payment_amounts.loc[:, ("X18", "X19", "X20")].mean(axis=1)
        engineered["payment_trend"] = _rowwise_trend(payment_amounts)
        engineered["payment_change_recent"] = payment_amounts["X18"] - payment_amounts["X19"]
        engineered["payment_cv"] = _safe_ratio(engineered["payment_std_v2"], engineered["payment_mean"].abs())

    if "ratios" in groups:
        for idx, bill_col in enumerate(BILL_AMOUNT_COLUMNS, start=1):
            pay_col = PAYMENT_AMOUNT_COLUMNS[idx - 1]
            engineered[f"payment_to_bill_ratio_m{idx}"] = _safe_ratio(payment_amounts[pay_col], bill_amounts[bill_col].abs())
            engineered[f"utilization_m{idx}"] = _safe_ratio(bill_amounts[bill_col], limit_bal.abs())
        ratio_cols = [f"payment_to_bill_ratio_m{idx}" for idx in range(1, 7)]
        util_cols = [f"utilization_m{idx}" for idx in range(1, 7)]
        engineered["payment_to_bill_ratio_mean"] = engineered.loc[:, ratio_cols].mean(axis=1)
        engineered["payment_to_bill_ratio_min"] = engineered.loc[:, ratio_cols].min(axis=1)
        engineered["payment_to_bill_ratio_recent"] = engineered["payment_to_bill_ratio_m1"]
        engineered["utilization_mean"] = engineered.loc[:, util_cols].mean(axis=1)
        engineered["utilization_max"] = engineered.loc[:, util_cols].max(axis=1)
        engineered["utilization_recent"] = engineered["utilization_m1"]
        engineered["utilization_trend"] = _rowwise_trend(engineered.loc[:, util_cols])
        engineered["utilization_cv"] = _safe_ratio(engineered.loc[:, util_cols].std(axis=1, ddof=0), engineered["utilization_mean"].abs())

    if "trends" in groups:
        utilization_frame = pd.DataFrame(
            {f"u{idx}": _safe_ratio(bill_amounts[bill_col], limit_bal.abs()) for idx, bill_col in enumerate(BILL_AMOUNT_COLUMNS, start=1)},
            index=engineered.index,
        )
        payment_ratio_recent = _safe_ratio(payment_amounts["X18"], bill_amounts["X12"].abs())
        engineered["bill_slope"] = _rowwise_trend(bill_amounts)
        engineered["payment_slope"] = _rowwise_trend(payment_amounts)
        engineered["utilization_slope"] = _rowwise_trend(utilization_frame)
        engineered["delay_slope"] = _rowwise_trend(pay_delay)
        engineered["payment_consistency"] = 1.0 - _safe_ratio(payment_amounts.std(axis=1, ddof=0), payment_amounts.mean(axis=1).abs()).clip(lower=0.0)
        engineered["bill_volatility"] = bill_amounts.std(axis=1, ddof=0)
        engineered["payment_volatility"] = payment_amounts.std(axis=1, ddof=0)
        engineered["delay_utilization_interaction"] = pay_delay["X6"] * utilization_frame["u1"]
        engineered["delay_payment_ratio_interaction"] = pay_delay["X6"] * payment_ratio_recent
        engineered["delay_count_utilization_interaction"] = pay_delay.gt(0).sum(axis=1).astype(float) * utilization_frame.mean(axis=1)
        engineered["severe_delay_low_payment_interaction"] = pay_delay.ge(3).any(axis=1).astype(float) * (1.0 - payment_ratio_recent.clip(upper=1.0))

    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    return engineered.fillna(0.0)


def engineer_credit_default_features(features: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible feature builder using all approved groups."""

    return build_behavioral_features(features)
