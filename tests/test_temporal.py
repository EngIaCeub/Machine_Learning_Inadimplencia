import pandas as pd

from credit_default.features.temporal import TEMPORAL_FEATURE_GROUPS, build_temporal_trajectory_features


def _frame() -> pd.DataFrame:
    row = {"X1": [1000], "X2": [1], "X3": [2], "X4": [1], "X5": [30]}
    row.update({f"X{i}": [0] for i in range(6, 12)})
    row.update({f"X{i}": [100] for i in range(12, 18)})
    row.update({f"X{i}": [50] for i in range(18, 24)})
    return pd.DataFrame(row)


def test_temporal_features_are_deterministic_safe_and_row_preserving():
    first = build_temporal_trajectory_features(_frame())
    second = build_temporal_trajectory_features(_frame())
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 1
    assert not first.isna().any().any()
    assert not first.select_dtypes(include="number").isin([float("inf"), float("-inf")]).any().any()
    assert set(TEMPORAL_FEATURE_GROUPS).issubset({"pay_trajectory", "bill_payment_trajectory", "shortfall_coverage", "utilization_interactions"})


def test_temporal_zero_denominators_are_safe():
    frame = _frame()
    frame.loc[0, "X1"] = 0
    frame.loc[0, "X12"] = 0
    frame.loc[0, "X18"] = 0
    result = build_temporal_trajectory_features(frame)
    assert not result.isna().any().any()
    assert not result.select_dtypes(include="number").isin([float("inf"), float("-inf")]).any().any()
