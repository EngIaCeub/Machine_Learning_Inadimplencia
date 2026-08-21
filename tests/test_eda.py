import pandas as pd

from credit_default.data.eda import build_eda_report


def test_eda_report_covers_gate_two_tables():
    rows = 30_000
    features = pd.DataFrame(
        {
            "LIMIT_BAL": [20_000] * 10_000 + [40_000] * 10_000 + [60_000] * 10_000,
            "AGE": [25] * 10_000 + [35] * 10_000 + [45] * 10_000,
            "SEX": [1] * 15_000 + [2] * 15_000,
            "EDUCATION": [1] * 10_000 + [2] * 10_000 + [6] * 10_000,
            "MARRIAGE": [1] * 15_000 + [2] * 15_000,
            "PAY_0": [0] * 10_000 + [2] * 10_000 + [9] * 10_000,
            "BILL_AMT1": [1_000] * rows,
            "PAY_AMT1": [500] * rows,
        }
    )
    features.loc[0, "LIMIT_BAL"] = None
    features = pd.concat([features, features.iloc[[1]]], ignore_index=True).iloc[:rows].copy()
    target = pd.Series([0] * 20_000 + [1] * 10_000, name="Y")

    report = build_eda_report(features, target)

    assert list(report.target_distribution.columns) == ["target", "count", "rate"]
    assert not report.descriptive_stats.empty
    assert report.missing_values.iloc[0]["column"] == "LIMIT_BAL"
    assert int(report.duplicate_summary.iloc[0]["duplicate_rows"]) >= 1
    assert "SEX" in report.default_rate_by_category
    assert "LIMIT_BAL" in report.default_rate_by_numeric_bin
    assert "EDUCATION" in report.inconsistent_values["column"].tolist()


def test_eda_report_is_objective_when_no_issues_exist():
    rows = 30_000
    features = pd.DataFrame(
        {
            "LIMIT_BAL": list(range(rows)),
            "AGE": [30 + (value % 20) for value in range(rows)],
            "SEX": [1 if value % 2 == 0 else 2 for value in range(rows)],
            "EDUCATION": [1 + (value % 4) for value in range(rows)],
            "MARRIAGE": [1 + (value % 3) for value in range(rows)],
            "PAY_0": [value % 4 for value in range(rows)],
            "BILL_AMT1": [1_000 + value for value in range(rows)],
            "PAY_AMT1": [500 + value for value in range(rows)],
        }
    )
    target = pd.Series([value % 2 for value in range(rows)], name="Y")

    report = build_eda_report(features, target)

    assert report.inconsistent_values.empty
    assert report.selected_numeric_columns
    assert report.selected_relation_columns
