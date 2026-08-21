import pandas as pd
import pytest

from credit_default.data.schema import build_eda_summary, build_schema_report, validate_basic_shape


def test_expected_shape_accepts_uci_scale():
    validate_basic_shape(30_000, 23)


def test_too_small_dataset_is_rejected():
    with pytest.raises(ValueError):
        validate_basic_shape(100, 23)


def test_schema_report_flags_missing_duplicates_and_id_leakage():
    features = pd.DataFrame(
        {
            "ID": [1, 1, 2, 3] + list(range(4, 30_000)),
            "LIMIT_BAL": [20_000, 20_000, None, 50_000] + [30_000] * 29_996,
            **{f"X{i}": [float(i)] * 30_000 for i in range(1, 22)},
        }
    )
    features.iloc[1] = features.iloc[0]
    target = pd.Series([0, 1] * 15_000, name="Y")

    report = build_schema_report(features, target)

    assert report.duplicate_rows >= 1
    assert report.missing_by_column["LIMIT_BAL"] == 1
    assert "missing values" in report.schema_issues
    assert "duplicate rows" in report.schema_issues
    assert report.leakage_risks == ("ID",)


def test_eda_summary_includes_target_balance_and_distribution_sections():
    features = pd.DataFrame(
        {
            "ID": range(30_000),
            "LIMIT_BAL": [20_000] * 15_000 + [40_000] * 15_000,
            "SEX": ["M"] * 15_000 + ["F"] * 15_000,
            **{f"X{i}": [float(i)] * 30_000 for i in range(1, 21)},
        }
    )
    target = pd.Series([0, 1] * 15_000, name="Y")

    summary = build_eda_summary(features, target)

    assert summary["shape"] == {"rows": 30_000, "features": 23}
    assert summary["target_balance"]["counts"] == {0: 15_000, 1: 15_000}
    assert "LIMIT_BAL" in summary["numeric_distributions"]
    assert summary["categorical_distributions"]["SEX"]["M"] == 15_000
