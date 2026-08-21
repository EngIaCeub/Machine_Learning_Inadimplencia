import pandas as pd
import pytest

from credit_default.features.credit_default_features import (
    ENGINEERED_FEATURE_COLUMNS,
    RAW_CATEGORICAL_COLUMNS,
    ROUND2_FEATURE_GROUPS,
    SEMANTIC_COLUMN_MAP,
    audit_credit_default_semantics,
    build_behavioral_features,
    engineer_credit_default_features,
)
from credit_default.features.preprocessing import fit_preprocessor, transform_features


def _raw_credit_default_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X1": [20_000, 0],
            "X2": [1, 2],
            "X3": [1, 6],
            "X4": [1, 0],
            "X5": [30, 45],
            "X6": [2, -1],
            "X7": [1, -1],
            "X8": [0, -2],
            "X9": [0, -2],
            "X10": [0, -2],
            "X11": [0, -2],
            "X12": [10_000, 0],
            "X13": [9_000, 0],
            "X14": [8_000, 0],
            "X15": [7_000, 0],
            "X16": [6_000, 0],
            "X17": [5_000, 0],
            "X18": [4_000, 0],
            "X19": [3_000, 0],
            "X20": [2_000, 0],
            "X21": [1_000, 0],
            "X22": [500, 0],
            "X23": [250, 0],
        }
    )


def test_semantic_audit_confirms_required_mapping():
    report = audit_credit_default_semantics(pd.Index(SEMANTIC_COLUMN_MAP))

    assert report.column_map["X1"] == "LIMIT_BAL"
    assert report.categorical_columns == RAW_CATEGORICAL_COLUMNS
    assert report.ordinal_delay_columns == ("X6", "X7", "X8", "X9", "X10", "X11")


def test_feature_engineering_is_deterministic_and_safe():
    raw = _raw_credit_default_frame()

    first = engineer_credit_default_features(raw)
    second = engineer_credit_default_features(raw)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "count_pay_delay_gt0"] == 2.0
    assert first.loc[0, "recent_pay_delay"] == 2
    assert first.loc[0, "pay_to_bill_ratio"] > 0
    assert first.loc[1, "pay_to_bill_ratio"] == 0.0
    assert first.loc[1, "utilization_proxy"] == 0.0


def test_feature_engineering_rejects_target_leakage_column():
    raw = _raw_credit_default_frame()
    raw["Y"] = [0, 1]

    with pytest.raises(ValueError, match="Target column"):
        engineer_credit_default_features(raw)


def test_engineered_feature_names_are_present():
    engineered = engineer_credit_default_features(_raw_credit_default_frame())

    for column in ENGINEERED_FEATURE_COLUMNS:
        assert column in engineered.columns


def test_round2_feature_groups_preserve_rows_and_avoid_invalid_numbers():
    engineered = build_behavioral_features(_raw_credit_default_frame(), enabled_groups=("pay", "bill", "payment", "ratios", "trends"))

    assert len(engineered) == len(_raw_credit_default_frame())
    assert not engineered.isna().any().any()
    assert not (engineered.select_dtypes(include=["number"]).isin([float("inf"), float("-inf")])).any().any()


def test_round2_feature_groups_are_incremental_and_deterministic():
    raw = _raw_credit_default_frame()

    first = build_behavioral_features(raw, enabled_groups=("pay", "ratios"))
    second = build_behavioral_features(raw, enabled_groups=("pay", "ratios"))

    pd.testing.assert_frame_equal(first, second)
    assert "pay_delay_count" in first.columns
    assert "payment_to_bill_ratio_m1" in first.columns
    assert "bill_mean" not in first.columns


def test_round2_feature_group_registry_is_non_empty():
    assert set(ROUND2_FEATURE_GROUPS) == {"round1", "pay", "bill", "payment", "ratios", "trends"}


def test_preprocessor_treats_x2_x3_x4_explicitly_and_fits_on_train_only():
    train = pd.concat([_raw_credit_default_frame()] * 3, ignore_index=True)
    validation = _raw_credit_default_frame().copy()
    validation.loc[:, "X3"] = [9, 7]

    bundle = fit_preprocessor(train)
    train_transformed = transform_features(bundle, train)
    validation_transformed = transform_features(bundle, validation)
    train_columns = train_transformed.columns.tolist()

    assert bundle.categorical_columns == ("X2", "X3", "X4")
    assert set(bundle.engineered_columns) == set(ENGINEERED_FEATURE_COLUMNS)
    assert "X3_9" not in validation_transformed.columns
    assert train_columns == validation_transformed.columns.tolist()
