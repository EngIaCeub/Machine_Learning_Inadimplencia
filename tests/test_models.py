import pandas as pd
import pytest
from sklearn.datasets import make_classification

from credit_default.data.split import make_splits
from credit_default.modeling import train as train_module
from credit_default.modeling.baseline import build_dummy_classifier
from credit_default.modeling.evaluate import (
    build_threshold_search_table,
    comparison_frame,
    evaluate_binary_classifier,
    evaluate_estimator,
    passes_academic_gates,
    summarize_precision_recall_tradeoff,
    tune_decision_threshold,
)
from credit_default.modeling.train import (
    build_xgboost_classifier,
    build_validation_variant_specs,
    _fit_estimator_with_optional_fallback,
    prepare_modeling_data,
    prepare_validation_modeling_data,
    run_modeling_workflow,
    run_validation_round_v2,
    train_candidates,
)
from credit_default.modeling.tune import (
    _resolve_xgboost_scale_pos_weight,
    select_promising_candidates,
    tune_model,
)


def _build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    features, target = make_classification(
        n_samples=600,
        n_features=23,
        n_informative=12,
        n_redundant=0,
        weights=[0.7, 0.3],
        random_state=42,
    )
    feature_frame = pd.DataFrame(features, columns=[f"X{i}" for i in range(1, features.shape[1] + 1)])
    feature_frame["X2"] = pd.cut(feature_frame["X2"], bins=2, labels=[1, 2]).astype(int)
    feature_frame["X3"] = pd.cut(feature_frame["X3"], bins=4, labels=[1, 2, 3, 4]).astype(int)
    feature_frame["X4"] = pd.cut(feature_frame["X4"], bins=3, labels=[1, 2, 3]).astype(int)
    target_series = pd.Series(target, name="Y")
    return feature_frame, target_series


def test_dummy_classifier_builder_returns_required_strategy():
    estimator = build_dummy_classifier()

    assert estimator.strategy == "prior"


def test_train_candidates_returns_required_model_ladder():
    features, target = _build_dataset()
    splits = make_splits(features, target)
    modeling_data = prepare_modeling_data(splits)

    trained = train_candidates(modeling_data)

    assert set(trained) == {
        "dummy_classifier",
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }


def test_validation_variant_specs_include_hgb_and_balanced_sample_weight():
    specs = build_validation_variant_specs()
    variant_names = {(spec.family_name, spec.variant_name) for spec in specs}

    assert ("hist_gradient_boosting", "none") in variant_names
    assert ("hist_gradient_boosting", "balanced_sample_weight") in variant_names
    assert ("xgboost", "baseline_1_0") in variant_names
    assert ("xgboost", "baseline_balanced") in variant_names


def test_build_xgboost_classifier_uses_cpu_first_defaults():
    estimator = build_xgboost_classifier()

    assert estimator.get_params()["objective"] == "binary:logistic"
    assert estimator.get_params()["eval_metric"] == "aucpr"
    assert estimator.get_params()["tree_method"] == "hist"
    assert estimator.get_params()["device"] == "cpu"


def test_xgboost_predict_proba_is_available_after_fit():
    features, target = _build_dataset()
    splits = make_splits(features, target)
    modeling_data = prepare_validation_modeling_data(splits)
    estimator = build_xgboost_classifier(n_estimators=50, max_depth=3)

    estimator.fit(modeling_data.X_train, modeling_data.y_train)
    probabilities = estimator.predict_proba(modeling_data.X_validation)

    assert probabilities.shape == (len(modeling_data.X_validation), 2)


def test_xgboost_builder_is_deterministic_for_same_seed():
    first = build_xgboost_classifier()
    second = build_xgboost_classifier()

    assert first.get_params()["random_state"] == second.get_params()["random_state"]


def test_resolve_xgboost_scale_pos_weight_uses_train_distribution_only():
    target = pd.Series([0, 0, 0, 0, 1, 1], name="Y")

    balanced = _resolve_xgboost_scale_pos_weight("balanced", target)
    light = _resolve_xgboost_scale_pos_weight("light", target)

    assert balanced == 2.0
    assert light == 1.5


def test_xgboost_cuda_fallback_rebuilds_cpu_estimator(monkeypatch: pytest.MonkeyPatch):
    features, target = _build_dataset()
    splits = make_splits(features, target)
    modeling_data = prepare_validation_modeling_data(splits)

    class FakeCudaEstimator:
        def __init__(self):
            self.device = "cuda"

        def fit(self, features, target, **kwargs):
            raise train_module.XGBoostError("no gpu")

        def get_params(self, deep: bool = True):
            return {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "tree_method": "hist",
                "device": "cuda",
                "random_state": 42,
                "n_jobs": 1,
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "min_child_weight": 1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "scale_pos_weight": 1.0,
            }

    class FakeCpuEstimator:
        def __init__(self, **params):
            self.params = params
            self.fitted = False

        def fit(self, features, target, **kwargs):
            self.fitted = True
            return self

    monkeypatch.setattr(train_module, "XGBClassifier", FakeCpuEstimator)

    fallback = _fit_estimator_with_optional_fallback(
        FakeCudaEstimator(),
        modeling_data.X_train,
        modeling_data.y_train,
    )

    assert isinstance(fallback, FakeCpuEstimator)
    assert fallback.params["device"] == "cpu"
    assert fallback.fitted is True


def test_tune_model_returns_best_params_for_promising_candidate():
    features, target = _build_dataset()
    splits = make_splits(features, target)
    modeling_data = prepare_modeling_data(splits)
    trained = train_candidates(modeling_data)

    result = tune_model(
        "logistic_regression",
        trained["logistic_regression"].estimator,
        modeling_data.X_train,
        modeling_data.y_train,
    )

    assert result.model_name == "logistic_regression"
    assert "C" in result.best_params
    assert result.refit_metric == "average_precision"


def test_tune_model_returns_numeric_scale_pos_weight_for_xgboost():
    features, target = _build_dataset()
    splits = make_splits(features, target)
    modeling_data = prepare_validation_modeling_data(splits)
    estimator = build_xgboost_classifier(n_estimators=50, max_depth=3)

    result = tune_model(
        "xgboost",
        estimator,
        modeling_data.X_train,
        modeling_data.y_train,
    )

    assert result.model_name == "xgboost"
    assert isinstance(result.best_params["scale_pos_weight"], float)
    assert hasattr(result.best_estimator, "predict_proba")


def test_threshold_tuning_prefers_recall_eligible_option():
    y_true = pd.Series([0, 0, 1, 1, 1, 0])
    scores = pd.Series([0.2, 0.4, 0.6, 0.7, 0.8, 0.3]).to_numpy()

    threshold, table = tune_decision_threshold(y_true, scores, min_recall=0.60)

    assert 0.0 <= threshold <= 1.0
    assert table["passes_recall_gate"].any()


def test_evaluate_binary_classifier_uses_positive_class_one_binary_f1():
    y_true = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.1, 0.7, 0.8, 0.4]).to_numpy()

    metrics = evaluate_binary_classifier(y_true, scores, threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision_positive"] == 0.5
    assert metrics["recall_positive"] == 0.5
    assert metrics["f1_binary"] == 0.5


def test_build_threshold_search_table_exposes_binary_macro_weighted_metrics():
    y_true = pd.Series([0, 0, 1, 1, 1, 0])
    scores = pd.Series([0.2, 0.4, 0.6, 0.7, 0.8, 0.3]).to_numpy()

    table = build_threshold_search_table(y_true, scores)

    assert {"f1_binary", "f1_macro", "f1_weighted", "valid_candidate"}.issubset(table.columns)
    assert table["roc_auc"].nunique() == 1


def test_precision_recall_summary_reports_max_f1_and_precision_gate():
    y_true = pd.Series([0, 0, 1, 1, 1, 0])
    scores = pd.Series([0.2, 0.4, 0.6, 0.7, 0.8, 0.3]).to_numpy()

    summary = summarize_precision_recall_tradeoff(y_true, scores)

    assert summary.average_precision > 0.0
    assert summary.max_f1 >= summary.f1_at_point_five
    assert summary.best_precision_at_recall_gate >= 0.0
    assert summary.constrained_max_f1 >= 0.0


def test_modeling_workflow_produces_validation_and_test_results():
    features, target = _build_dataset()
    splits = make_splits(features, target)

    result = run_modeling_workflow(splits)

    assert result.base_validation_results
    assert result.promising_candidates
    assert len(result.promising_candidates) == 3
    assert not result.champion_selection_comparison.empty
    assert result.champion_validation_result.split_name == "validation"
    assert result.champion_test_result.split_name == "test"
    assert result.champion_validation_result.threshold == result.champion_test_result.threshold
    assert set(result.champion_test_result.metrics) == {
        "roc_auc",
        "f1",
        "recall",
        "precision",
        "accuracy",
    }


def test_comparison_and_promising_selection_are_objective():
    features, target = _build_dataset()
    splits = make_splits(features, target)
    modeling_data = prepare_modeling_data(splits)
    trained = train_candidates(modeling_data)

    validation_results = [
        evaluate_estimator(name, trained_model.estimator, modeling_data.X_validation, modeling_data.y_validation)
        for name, trained_model in trained.items()
    ]
    table = comparison_frame(validation_results)
    promising = select_promising_candidates(validation_results)

    assert not table.empty
    assert len(promising) == 3
    assert "dummy_classifier" not in promising


def test_validation_round_v2_is_validation_only_and_freezes_threshold_if_ready():
    features, target = _build_dataset()
    splits = make_splits(features, target)

    result = run_validation_round_v2(splits)

    assert result.base_results is not None
    assert result.tuned_results is not None
    assert result.promising_families
    assert result.status in {"READY_FOR_FINAL_EVALUATION", "NEEDS_GATE_REVIEW"}
    assert result.frozen_model_family is not None
    assert result.frozen_threshold is not None
    assert result.prepared_data.X_validation.shape[0] == len(splits.y_validation)
    assert "hist_gradient_boosting" in result.promising_families
    assert "xgboost" in result.promising_families
