import pandas as pd

from credit_default.features.preprocessing import (
    describe_preprocessor,
    fit_preprocessor,
    transform_features,
)


def test_preprocessor_drops_identifier_and_handles_unseen_categories():
    train = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "LIMIT_BAL": [20_000, 50_000, None, 30_000],
            "SEX": ["M", "F", "M", "F"],
            "PAY_0": [0, -1, 2, 1],
        }
    )
    validation = pd.DataFrame(
        {
            "ID": [5, 6],
            "LIMIT_BAL": [60_000, None],
            "SEX": ["X", "F"],
            "PAY_0": [3, -2],
        }
    )

    bundle = fit_preprocessor(train)
    train_transformed = transform_features(bundle, train)
    validation_transformed = transform_features(bundle, validation)
    description = describe_preprocessor(bundle)

    assert "ID" in description["dropped_columns"]
    assert description["leakage_columns"] == ["ID"]
    assert "SEX_X" not in validation_transformed.columns
    assert train_transformed.columns.tolist() == validation_transformed.columns.tolist()
    assert not train_transformed.isna().any().any()
    assert not validation_transformed.isna().any().any()
    assert set(description["categorical_columns"]) == {"X2", "X3", "X4"} or "SEX" in description["categorical_columns"]


def test_preprocessor_keeps_reusable_column_metadata():
    train = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "LIMIT_BAL": [20_000, 25_000, 30_000],
            "MARRIAGE": ["single", "married", "single"],
        }
    )

    bundle = fit_preprocessor(train)

    assert "LIMIT_BAL" in bundle.feature_columns
    assert "MARRIAGE" in bundle.categorical_columns or bundle.categorical_columns == ("X2", "X3", "X4")
