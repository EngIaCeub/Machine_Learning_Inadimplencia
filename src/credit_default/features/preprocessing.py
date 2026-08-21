"""Reusable preprocessing for S02."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_default.features.credit_default_features import (
    ENGINEERED_FEATURE_COLUMNS,
    RAW_CATEGORICAL_COLUMNS,
    build_behavioral_features,
    engineer_credit_default_features,
    has_credit_default_schema,
)

IDENTIFIER_PATTERN = re.compile(r"(^id$|_id$)", re.IGNORECASE)


@dataclass(frozen=True)
class PreprocessingBundle:
    transformer: ColumnTransformer
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    engineered_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    leakage_columns: tuple[str, ...]


def detect_identifier_columns(columns: list[str] | tuple[str, ...] | pd.Index) -> list[str]:
    return [column for column in columns if IDENTIFIER_PATTERN.search(str(column))]


def _make_one_hot_encoder() -> OneHotEncoder:
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    features: pd.DataFrame,
    drop_columns: list[str] | tuple[str, ...] | None = None,
    feature_builder: Any | None = None,
) -> PreprocessingBundle:
    """Build an unfitted sklearn-compatible preprocessing graph."""

    explicit_drops = list(drop_columns or [])
    builder = feature_builder or engineer_credit_default_features
    modeled_features = builder(features) if has_credit_default_schema(features.columns) else features.copy()
    leakage_columns = detect_identifier_columns(modeled_features.columns)
    dropped_columns = tuple(dict.fromkeys([*explicit_drops, *leakage_columns]))

    feature_columns = tuple(column for column in modeled_features.columns if column not in dropped_columns)
    selected_features = modeled_features.loc[:, feature_columns]
    if has_credit_default_schema(features.columns):
        categorical_columns = tuple(column for column in feature_columns if column in RAW_CATEGORICAL_COLUMNS)
        numeric_columns = tuple(column for column in feature_columns if column not in categorical_columns)
    else:
        numeric_columns = tuple(selected_features.select_dtypes(include=["number", "bool"]).columns.tolist())
        categorical_columns = tuple(
            column for column in selected_features.columns if column not in numeric_columns
        )

    transformers: list[tuple[str, Any, list[str] | tuple[str, ...]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_columns),
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                list(categorical_columns),
            )
        )

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return PreprocessingBundle(
        transformer=transformer,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        engineered_columns=tuple(column for column in ENGINEERED_FEATURE_COLUMNS if column in feature_columns),
        dropped_columns=dropped_columns,
        leakage_columns=tuple(leakage_columns),
    )


def fit_preprocessor(
    features_train: pd.DataFrame,
    drop_columns: list[str] | tuple[str, ...] | None = None,
    feature_builder: Any | None = None,
) -> PreprocessingBundle:
    """Fit preprocessing on training data only."""

    builder = feature_builder or engineer_credit_default_features
    bundle = build_preprocessor(features_train, drop_columns=drop_columns, feature_builder=builder)
    transformed_train = (
        builder(features_train)
        if has_credit_default_schema(features_train.columns)
        else features_train.copy()
    )
    bundle.transformer.fit(transformed_train.loc[:, list(bundle.feature_columns)])
    return bundle


def transform_features(
    bundle: PreprocessingBundle,
    features: pd.DataFrame,
    feature_builder: Any | None = None,
) -> pd.DataFrame:
    """Apply a fitted preprocessor and preserve row index for split tracking."""

    builder = feature_builder or engineer_credit_default_features
    engineered = builder(features) if has_credit_default_schema(features.columns) else features.copy()
    transformed = bundle.transformer.transform(engineered.loc[:, list(bundle.feature_columns)])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    output_columns = bundle.transformer.get_feature_names_out().tolist()
    return pd.DataFrame(transformed, columns=output_columns, index=features.index)


def describe_preprocessor(bundle: PreprocessingBundle) -> dict[str, Any]:
    """Return gate-friendly preprocessing metadata."""

    return {
        "feature_columns": list(bundle.feature_columns),
        "numeric_columns": list(bundle.numeric_columns),
        "categorical_columns": list(bundle.categorical_columns),
        "engineered_columns": list(bundle.engineered_columns),
        "dropped_columns": list(bundle.dropped_columns),
        "leakage_columns": list(bundle.leakage_columns),
        "steps": {
            "numeric": ["median_imputer", "standard_scaler"] if bundle.numeric_columns else [],
            "categorical": ["most_frequent_imputer", "one_hot_encoder"]
            if bundle.categorical_columns
            else [],
        },
    }
