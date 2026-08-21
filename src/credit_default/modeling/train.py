"""Candidate training and S03 workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from credit_default.config import get_project_config
from credit_default.data.split import DatasetSplits
from credit_default.features.credit_default_features import audit_credit_default_semantics
from credit_default.features.preprocessing import (
    PreprocessingBundle,
    fit_preprocessor,
    transform_features,
)
from credit_default.modeling.baseline import build_dummy_classifier
from credit_default.modeling.evaluate import (
    EvaluationResult,
    ValidationThresholdSummary,
    comparison_frame,
    evaluate_estimator,
    get_positive_class_scores,
    select_best_result,
    summarize_precision_recall_tradeoff,
    tune_decision_threshold,
)
from credit_default.modeling.tune import TuningResult, select_promising_candidates, tune_model


@dataclass(frozen=True)
class PreparedModelingData:
    preprocessor_bundle: PreprocessingBundle
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class PreparedValidationData:
    preprocessor_bundle: PreprocessingBundle
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series


@dataclass(frozen=True)
class ModelVariantSpec:
    family_name: str
    variant_name: str
    estimator: Any
    sample_weight_mode: str | None = None


@dataclass(frozen=True)
class TrainedModel:
    model_name: str
    estimator: object
    tuned: bool


@dataclass(frozen=True)
class ModelingWorkflowResult:
    prepared_data: PreparedModelingData
    base_validation_results: list[EvaluationResult]
    tuning_results: dict[str, TuningResult]
    tuned_validation_results: list[EvaluationResult]
    validation_comparison: pd.DataFrame
    champion_selection_comparison: pd.DataFrame
    promising_candidates: tuple[str, ...]
    champion_validation_result: EvaluationResult
    champion_test_result: EvaluationResult
    threshold_search: pd.DataFrame


@dataclass(frozen=True)
class ValidationExperimentResult:
    semantic_audit: dict[str, Any]
    prepared_data: PreparedValidationData
    base_results: pd.DataFrame
    tuned_results: pd.DataFrame
    promising_families: tuple[str, ...]
    frozen_model_family: str | None
    frozen_model_variant: str | None
    frozen_threshold: float | None
    frozen_hyperparameters: dict[str, Any] | None
    frozen_feature_columns: tuple[str, ...] | None
    status: str
    recommend_hist_gradient_boosting: bool


MANUAL_LIGHT_CLASS_WEIGHT = {0: 1.0, 1: 2.0}


def prepare_modeling_data(splits: DatasetSplits) -> PreparedModelingData:
    """Fit preprocessing on train only and transform all S02 splits."""

    preprocessor_bundle = fit_preprocessor(splits.X_train)
    return PreparedModelingData(
        preprocessor_bundle=preprocessor_bundle,
        X_train=transform_features(preprocessor_bundle, splits.X_train),
        X_validation=transform_features(preprocessor_bundle, splits.X_validation),
        X_test=transform_features(preprocessor_bundle, splits.X_test),
        y_train=splits.y_train.copy(),
        y_validation=splits.y_validation.copy(),
        y_test=splits.y_test.copy(),
    )


def prepare_validation_modeling_data(splits: DatasetSplits) -> PreparedValidationData:
    """Fit preprocessing on train only and transform train/validation only."""

    preprocessor_bundle = fit_preprocessor(splits.X_train)
    return PreparedValidationData(
        preprocessor_bundle=preprocessor_bundle,
        X_train=transform_features(preprocessor_bundle, splits.X_train),
        X_validation=transform_features(preprocessor_bundle, splits.X_validation),
        y_train=splits.y_train.copy(),
        y_validation=splits.y_validation.copy(),
    )


def build_candidate_estimators() -> dict[str, object]:
    """Build the required S03 model ladder using the approved S02 pipeline."""

    random_seed = get_project_config().random_seed
    return {
        "dummy_classifier": build_dummy_classifier(random_seed=random_seed),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=random_seed,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            min_samples_leaf=50,
            random_state=random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_seed,
        ),
    }


def build_validation_variant_specs() -> list[ModelVariantSpec]:
    """Build controlled imbalance variants for validation-only S03-v2."""

    random_seed = get_project_config().random_seed
    return [
        ModelVariantSpec(
            family_name="dummy_classifier",
            variant_name="prior",
            estimator=build_dummy_classifier(random_seed=random_seed),
        ),
        ModelVariantSpec(
            family_name="logistic_regression",
            variant_name="none",
            estimator=LogisticRegression(
                class_weight=None,
                max_iter=2_000,
                solver="liblinear",
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="logistic_regression",
            variant_name="balanced",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                solver="liblinear",
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="logistic_regression",
            variant_name="manual_light",
            estimator=LogisticRegression(
                class_weight=MANUAL_LIGHT_CLASS_WEIGHT,
                max_iter=2_000,
                solver="liblinear",
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="decision_tree",
            variant_name="none",
            estimator=DecisionTreeClassifier(
                class_weight=None,
                min_samples_leaf=50,
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="decision_tree",
            variant_name="balanced",
            estimator=DecisionTreeClassifier(
                class_weight="balanced",
                min_samples_leaf=50,
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="decision_tree",
            variant_name="manual_light",
            estimator=DecisionTreeClassifier(
                class_weight=MANUAL_LIGHT_CLASS_WEIGHT,
                min_samples_leaf=50,
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="random_forest",
            variant_name="none",
            estimator=RandomForestClassifier(
                n_estimators=300,
                class_weight=None,
                n_jobs=-1,
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="random_forest",
            variant_name="balanced",
            estimator=RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="random_forest",
            variant_name="balanced_subsample",
            estimator=RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="random_forest",
            variant_name="manual_light",
            estimator=RandomForestClassifier(
                n_estimators=300,
                class_weight=MANUAL_LIGHT_CLASS_WEIGHT,
                n_jobs=-1,
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="hist_gradient_boosting",
            variant_name="none",
            estimator=HistGradientBoostingClassifier(
                random_state=random_seed,
            ),
        ),
        ModelVariantSpec(
            family_name="hist_gradient_boosting",
            variant_name="balanced_sample_weight",
            estimator=HistGradientBoostingClassifier(
                random_state=random_seed,
            ),
            sample_weight_mode="balanced",
        ),
        ModelVariantSpec(
            family_name="xgboost",
            variant_name="baseline_1_0",
            estimator=build_xgboost_classifier(scale_pos_weight=1.0),
        ),
        ModelVariantSpec(
            family_name="xgboost",
            variant_name="baseline_light",
            estimator=build_xgboost_classifier(
                scale_pos_weight=1.5,
            ),
        ),
        ModelVariantSpec(
            family_name="xgboost",
            variant_name="baseline_balanced",
            estimator=build_xgboost_classifier(
                scale_pos_weight="balanced",
            ),
        ),
    ]


def build_xgboost_classifier(
    *,
    scale_pos_weight: float | str = 1.0,
    device: str = "cpu",
    random_seed: int | None = None,
    **overrides: Any,
) -> XGBClassifier:
    """Build CPU-first XGBoost with optional CUDA acceleration."""

    resolved_seed = random_seed if random_seed is not None else get_project_config().random_seed
    resolved_scale = scale_pos_weight if isinstance(scale_pos_weight, float) else 1.0
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": device,
        "random_state": resolved_seed,
        "n_jobs": 1,
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
        "scale_pos_weight": resolved_scale,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def train_candidates(modeling_data: PreparedModelingData) -> dict[str, TrainedModel]:
    """Fit baseline and required candidate models on transformed training data."""

    trained_models: dict[str, TrainedModel] = {}
    for model_name, estimator in build_candidate_estimators().items():
        estimator.fit(modeling_data.X_train, modeling_data.y_train)
        trained_models[model_name] = TrainedModel(
            model_name=model_name,
            estimator=estimator,
            tuned=False,
        )
    return trained_models


def _validation_result_row(
    family_name: str,
    variant_name: str,
    estimator: Any,
    features_validation: pd.DataFrame,
    target_validation: pd.Series,
    tuned: bool,
) -> tuple[dict[str, Any], ValidationThresholdSummary]:
    scores = get_positive_class_scores(estimator, features_validation)
    metrics_at_point_five, _, _ = (
        evaluate_estimator(
            model_name=family_name,
            estimator=estimator,
            features=features_validation,
            target=target_validation,
            threshold=0.5,
            split_name="validation",
            tuned=tuned,
        ).metrics,
        None,
        None,
    )
    pr_summary = summarize_precision_recall_tradeoff(target_validation, scores)
    return (
        {
            "model_family": family_name,
            "variant_name": variant_name,
            "tuned": tuned,
            "roc_auc": metrics_at_point_five["roc_auc"],
            "average_precision": pr_summary.average_precision,
            "f1_at_0_5": pr_summary.f1_at_point_five,
            "max_f1": pr_summary.max_f1,
            "max_f1_threshold": pr_summary.max_f1_threshold,
            "recall_at_max_f1": pr_summary.recall_at_max_f1,
            "precision_at_max_f1": pr_summary.precision_at_max_f1,
            "constrained_max_f1": pr_summary.constrained_max_f1,
            "constrained_threshold": pr_summary.constrained_max_f1_threshold,
            "constrained_recall": pr_summary.constrained_recall,
            "constrained_precision": pr_summary.constrained_precision,
            "best_precision_at_recall_ge_0_60": pr_summary.best_precision_at_recall_gate,
            "threshold_at_best_precision_recall_ge_0_60": pr_summary.threshold_at_best_precision_recall_gate,
            "passes_validation_gate": (
                pr_summary.constrained_max_f1 >= 0.65 and pr_summary.constrained_recall >= 0.60
            ),
        },
        pr_summary,
    )


def _select_promising_families(base_results: pd.DataFrame, max_families: int = 4) -> tuple[str, ...]:
    family_best = (
        base_results.sort_values(
            ["constrained_max_f1", "average_precision", "roc_auc", "constrained_precision"],
            ascending=[False, False, False, False],
        )
        .groupby("model_family", as_index=False)
        .head(1)
    )
    family_best = family_best.loc[family_best["model_family"] != "dummy_classifier"]
    family_best = family_best.sort_values(
        ["constrained_max_f1", "average_precision", "roc_auc", "constrained_precision"],
        ascending=[False, False, False, False],
    )
    return tuple(family_best["model_family"].head(max_families).tolist())


def _build_sample_weight(mode: str | None, target: pd.Series) -> pd.Series | None:
    if mode is None:
        return None
    if mode == "balanced":
        return pd.Series(compute_sample_weight(class_weight="balanced", y=target), index=target.index)
    raise ValueError(f"Unknown sample weight mode: {mode}.")


def _resolve_scale_pos_weight(value: float | str, target: pd.Series) -> float:
    if isinstance(value, float):
        return value
    positive_count = int((target == 1).sum())
    negative_count = int((target == 0).sum())
    balanced = negative_count / positive_count
    if value == "balanced":
        return float(balanced)
    if value == "light":
        return float((1.0 + balanced) / 2.0)
    raise ValueError(f"Unknown scale_pos_weight strategy: {value}.")


def _fit_estimator_with_optional_fallback(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    sample_weight: pd.Series | None = None,
) -> Any:
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.to_numpy()
    try:
        estimator.fit(features, target, **fit_kwargs)
        return estimator
    except XGBoostError:
        if getattr(estimator, "device", None) != "cuda":
            raise
        fallback = estimator.get_params()
        fallback["device"] = "cpu"
        cpu_estimator = XGBClassifier(**fallback)
        cpu_estimator.fit(features, target, **fit_kwargs)
        return cpu_estimator


def run_validation_round_v2(splits: DatasetSplits) -> ValidationExperimentResult:
    """Run S03-v2 on validation only."""

    prepared = prepare_validation_modeling_data(splits)
    semantic_audit = audit_credit_default_semantics(splits.X_train.columns)
    base_rows: list[dict[str, Any]] = []
    variant_specs = build_validation_variant_specs()
    trained_estimators: dict[tuple[str, str], Any] = {}
    sample_weight_modes: dict[tuple[str, str], str | None] = {}
    for spec in variant_specs:
        estimator = spec.estimator
        if spec.family_name == "xgboost":
            estimator = build_xgboost_classifier(
                scale_pos_weight=_resolve_scale_pos_weight(
                    estimator.get_params()["scale_pos_weight"],
                    prepared.y_train,
                ),
                device=estimator.get_params().get("device", "cpu"),
                random_seed=get_project_config().random_seed,
                n_estimators=estimator.get_params()["n_estimators"],
                max_depth=estimator.get_params()["max_depth"],
                learning_rate=estimator.get_params()["learning_rate"],
                min_child_weight=estimator.get_params()["min_child_weight"],
                subsample=estimator.get_params()["subsample"],
                colsample_bytree=estimator.get_params()["colsample_bytree"],
                reg_lambda=estimator.get_params()["reg_lambda"],
                reg_alpha=estimator.get_params()["reg_alpha"],
            )
        sample_weight = _build_sample_weight(spec.sample_weight_mode, prepared.y_train)
        fitted_estimator = _fit_estimator_with_optional_fallback(
            estimator,
            prepared.X_train,
            prepared.y_train,
            sample_weight=sample_weight,
        )
        trained_estimators[(spec.family_name, spec.variant_name)] = fitted_estimator
        sample_weight_modes[(spec.family_name, spec.variant_name)] = spec.sample_weight_mode
        row, _ = _validation_result_row(
            family_name=spec.family_name,
            variant_name=spec.variant_name,
            estimator=fitted_estimator,
            features_validation=prepared.X_validation,
            target_validation=prepared.y_validation,
            tuned=False,
        )
        base_rows.append(row)

    base_results = pd.DataFrame(base_rows).sort_values(
        ["constrained_max_f1", "average_precision", "roc_auc", "constrained_precision"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    promising_families = _select_promising_families(base_results)
    tuned_rows: list[dict[str, Any]] = []
    for family_name in promising_families:
        best_base_row = (
            base_results.loc[base_results["model_family"] == family_name]
            .sort_values(
                ["constrained_max_f1", "average_precision", "roc_auc"],
                ascending=[False, False, False],
            )
            .iloc[0]
        )
        best_base_estimator = trained_estimators[(family_name, best_base_row["variant_name"])]
        sample_weight_mode = sample_weight_modes[(family_name, best_base_row["variant_name"])]
        sample_weight = _build_sample_weight(sample_weight_mode, prepared.y_train)
        if family_name == "xgboost":
            best_base_params = best_base_estimator.get_params()
            best_base_estimator = build_xgboost_classifier(
                scale_pos_weight=best_base_params["scale_pos_weight"],
                device=best_base_params.get("device", "cpu"),
                random_seed=get_project_config().random_seed,
                n_estimators=best_base_params["n_estimators"],
                max_depth=best_base_params["max_depth"],
                learning_rate=best_base_params["learning_rate"],
                min_child_weight=best_base_params["min_child_weight"],
                subsample=best_base_params["subsample"],
                colsample_bytree=best_base_params["colsample_bytree"],
                reg_lambda=best_base_params["reg_lambda"],
                reg_alpha=best_base_params["reg_alpha"],
            )
        tuning_result = tune_model(
            model_name=family_name,
            estimator=best_base_estimator,
            features=prepared.X_train,
            target=prepared.y_train,
            sample_weight=sample_weight,
        )
        tuned_row, _ = _validation_result_row(
            family_name=family_name,
            variant_name="tuned",
            estimator=tuning_result.best_estimator,
            features_validation=prepared.X_validation,
            target_validation=prepared.y_validation,
            tuned=True,
        )
        tuned_row["best_params"] = tuning_result.best_params
        tuned_row["refit_metric"] = tuning_result.refit_metric
        tuned_row["sample_weight_mode"] = sample_weight_mode
        tuned_rows.append(tuned_row)

    tuned_results = pd.DataFrame(tuned_rows).sort_values(
        ["constrained_max_f1", "average_precision", "roc_auc", "constrained_precision"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    combined_results = pd.concat([base_results, tuned_results], ignore_index=True, sort=False)
    model_only_results = combined_results.loc[
        combined_results["model_family"] != "dummy_classifier"
    ].copy()
    eligible = model_only_results.loc[model_only_results["recall_at_max_f1"] >= 0.60].copy()
    eligible = eligible.loc[eligible["constrained_recall"] >= 0.60].copy()
    if not eligible.empty:
        selected = eligible.sort_values(
            ["constrained_max_f1", "average_precision", "roc_auc", "constrained_precision"],
            ascending=[False, False, False, False],
        ).iloc[0]
    else:
        selected = model_only_results.sort_values(
            ["constrained_max_f1", "average_precision", "roc_auc", "constrained_precision"],
            ascending=[False, False, False, False],
        ).iloc[0]

    status = (
        "READY_FOR_FINAL_EVALUATION"
        if float(selected["constrained_max_f1"]) >= 0.65 and float(selected["constrained_recall"]) >= 0.60
        else "NEEDS_GATE_REVIEW"
    )

    return ValidationExperimentResult(
        semantic_audit={
            "column_map": semantic_audit.column_map,
            "categorical_columns": list(semantic_audit.categorical_columns),
            "ordinal_delay_columns": list(semantic_audit.ordinal_delay_columns),
            "bill_columns": list(semantic_audit.bill_columns),
            "payment_columns": list(semantic_audit.payment_columns),
        },
        prepared_data=prepared,
        base_results=base_results,
        tuned_results=tuned_results,
        promising_families=promising_families,
        frozen_model_family=str(selected["model_family"]),
        frozen_model_variant=str(selected["variant_name"]),
        frozen_threshold=float(selected["constrained_threshold"]),
        frozen_hyperparameters=(
            dict(selected["best_params"]) if "best_params" in selected and isinstance(selected["best_params"], dict) else None
        ),
        frozen_feature_columns=prepared.preprocessor_bundle.feature_columns,
        status=status,
        recommend_hist_gradient_boosting=(status == "NEEDS_GATE_REVIEW"),
    )


def run_modeling_workflow(splits: DatasetSplits) -> ModelingWorkflowResult:
    """Execute S03 without touching test data until champion+threshold are frozen."""

    modeling_data = prepare_modeling_data(splits)
    trained_models = train_candidates(modeling_data)

    base_validation_results = [
        evaluate_estimator(
            model_name=model_name,
            estimator=trained.estimator,
            features=modeling_data.X_validation,
            target=modeling_data.y_validation,
            threshold=0.5,
            split_name="validation",
            tuned=trained.tuned,
        )
        for model_name, trained in trained_models.items()
    ]

    promising_candidates = select_promising_candidates(base_validation_results)
    tuning_results: dict[str, TuningResult] = {}
    tuned_validation_results: list[EvaluationResult] = []
    for model_name in promising_candidates:
        tuning_result = tune_model(
            model_name=model_name,
            estimator=trained_models[model_name].estimator,
            features=modeling_data.X_train,
            target=modeling_data.y_train,
        )
        tuning_results[model_name] = tuning_result
        tuned_validation_results.append(
            evaluate_estimator(
                model_name=model_name,
                estimator=tuning_result.best_estimator,
                features=modeling_data.X_validation,
                target=modeling_data.y_validation,
                threshold=0.5,
                split_name="validation",
                tuned=True,
            )
        )

    all_validation_results = [
        result for result in base_validation_results if result.model_name not in tuning_results
    ] + tuned_validation_results
    validation_comparison = comparison_frame(all_validation_results)

    threshold_optimized_results: list[EvaluationResult] = []
    threshold_tables: dict[str, pd.DataFrame] = {}
    for result in all_validation_results:
        if result.model_name == "dummy_classifier":
            continue
        candidate_estimator = (
            tuning_results[result.model_name].best_estimator
            if result.tuned
            else trained_models[result.model_name].estimator
        )
        candidate_scores = get_positive_class_scores(candidate_estimator, modeling_data.X_validation)
        candidate_threshold, candidate_threshold_table = tune_decision_threshold(
            modeling_data.y_validation,
            candidate_scores,
        )
        threshold_tables[result.model_name] = candidate_threshold_table
        threshold_optimized_results.append(
            evaluate_estimator(
                model_name=result.model_name,
                estimator=candidate_estimator,
                features=modeling_data.X_validation,
                target=modeling_data.y_validation,
                threshold=candidate_threshold,
                split_name="validation",
                tuned=result.tuned,
            )
        )

    champion_selection_comparison = comparison_frame(threshold_optimized_results)
    champion_validation_result = select_best_result(threshold_optimized_results)
    champion_estimator = (
        tuning_results[champion_validation_result.model_name].best_estimator
        if champion_validation_result.tuned
        else trained_models[champion_validation_result.model_name].estimator
    )
    threshold = champion_validation_result.threshold
    threshold_search = threshold_tables[champion_validation_result.model_name]
    frozen_test_result = evaluate_estimator(
        model_name=champion_validation_result.model_name,
        estimator=champion_estimator,
        features=modeling_data.X_test,
        target=modeling_data.y_test,
        threshold=threshold,
        split_name="test",
        tuned=champion_validation_result.tuned,
    )

    return ModelingWorkflowResult(
        prepared_data=modeling_data,
        base_validation_results=base_validation_results,
        tuning_results=tuning_results,
        tuned_validation_results=tuned_validation_results,
        validation_comparison=validation_comparison,
        champion_selection_comparison=champion_selection_comparison,
        promising_candidates=promising_candidates,
        champion_validation_result=champion_validation_result,
        champion_test_result=frozen_test_result,
        threshold_search=threshold_search,
    )
