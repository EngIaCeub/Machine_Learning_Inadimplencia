import pandas as pd
import pytest

from credit_default.data.split import make_splits, summarize_splits, validate_distinct_splits


def _build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rows = 100
    features = pd.DataFrame(
        {
            "ID": range(rows),
            "LIMIT_BAL": [20_000 + value for value in range(rows)],
            "SEX": ["M" if value % 2 == 0 else "F" for value in range(rows)],
            **{f"X{i}": [float(i + value) for value in range(rows)] for i in range(1, 21)},
        },
        index=range(1_000, 1_000 + rows),
    )
    target = pd.Series(([0, 1] * (rows // 2)), index=features.index, name="Y")
    return features, target


def test_make_splits_is_deterministic_and_stratified():
    features, target = _build_dataset()

    first = make_splits(features, target)
    second = make_splits(features, target)

    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert first.X_validation.index.tolist() == second.X_validation.index.tolist()
    assert first.X_test.index.tolist() == second.X_test.index.tolist()

    summary = summarize_splits(first)
    assert summary["train"]["rows"] == 70
    assert summary["validation"]["rows"] == 15
    assert summary["test"]["rows"] == 15
    assert summary["train"]["positive_rate"] == 0.5
    assert summary["validation"]["positive_rate"] == pytest.approx(0.5, abs=0.05)
    assert summary["test"]["positive_rate"] == pytest.approx(0.5, abs=0.05)


def test_split_indices_do_not_overlap():
    features, target = _build_dataset()

    splits = make_splits(features, target)

    validate_distinct_splits(splits)
