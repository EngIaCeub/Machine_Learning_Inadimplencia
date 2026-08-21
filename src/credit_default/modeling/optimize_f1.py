"""Validation-only binary-F1 optimization workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.utils.class_weight import compute_sample_weight

from credit_default.config import get_project_config
from credit_default.data.split import DatasetSplits, make_splits
from credit_default.modeling.evaluate import (
    REQUIRED_METRIC_GATES,
    build_threshold_search_table,
    evaluate_binary_classifier,
    get_positive_class_scores,
    select_best_threshold_row,
)
from credit_default.modeling.train import (
    _fit_estimator_with_optional_fallback,
    _resolve_scale_pos_weight,
    build_xgboost_classifier,
    prepare_validation_modeling_data,
)
from credit_default.modeling.tune import tune_model


matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "experiments"


@dataclass(frozen=True)
class ExperimentRecord:
    experiment: str
    model: str
    strategy: str
    threshold: float
    precision: float
    recall: float
    f1_binary: float
    macro_f1: float
    weighted_f1: float
    roc_auc: float
    accuracy: float
    seed: int
    status: str
    class_weighting: str
    hyperparameters: dict[str, Any]
    timestamp: str


@dataclass(frozen=True)
class OptimizationRunResult:
    audit_summary: pd.DataFrame
    baseline: ExperimentRecord
    best_result: ExperimentRecord
    best_metrics: dict[str, Any]
    experiment_records: pd.DataFrame
    threshold_search: pd.DataFrame
    model_comparison: pd.DataFrame
    tuning_results: pd.DataFrame
    artifact_dir: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _status_from_metrics(metrics: dict[str, Any]) -> str:
    official_f1 = float(metrics.get("official_a1_f1", metrics.get("f1_macro", metrics["f1_binary"])))
    if (
        float(metrics["roc_auc"]) >= REQUIRED_METRIC_GATES["roc_auc"]
        and float(metrics["recall_positive"]) >= REQUIRED_METRIC_GATES["recall"]
        and official_f1 >= REQUIRED_METRIC_GATES["f1"]
    ):
        return "PASS"
    if (
        float(metrics["roc_auc"]) >= REQUIRED_METRIC_GATES["roc_auc"]
        and float(metrics["recall_positive"]) >= REQUIRED_METRIC_GATES["recall"]
    ):
        return "VALID_BUT_F1_LOW"
    return "FAIL"


def _make_record(
    *,
    experiment: str,
    model: str,
    strategy: str,
    metrics: dict[str, Any],
    seed: int,
    class_weighting: str,
    hyperparameters: dict[str, Any],
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment=experiment,
        model=model,
        strategy=strategy,
        threshold=float(metrics["threshold"]),
        precision=float(metrics["precision_positive"]),
        recall=float(metrics["recall_positive"]),
        f1_binary=float(metrics["f1_binary"]),
        macro_f1=float(metrics["f1_macro"]),
        weighted_f1=float(metrics["f1_weighted"]),
        roc_auc=float(metrics["roc_auc"]),
        accuracy=float(metrics["accuracy"]),
        seed=seed,
        status=_status_from_metrics(metrics),
        class_weighting=class_weighting,
        hyperparameters=hyperparameters,
        timestamp=_utc_now_iso(),
    )


def _evaluate_estimator_on_validation(
    *,
    estimator: Any,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    scores = get_positive_class_scores(estimator, X_validation)
    threshold_table = build_threshold_search_table(y_validation, scores)
    best_row = select_best_threshold_row(threshold_table)
    metrics = best_row.to_dict()
    return metrics, threshold_table, scores


def _fit_with_optional_sample_weight(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series | None = None,
) -> Any:
    return _fit_estimator_with_optional_fallback(
        estimator,
        X_train,
        y_train,
        sample_weight=sample_weight,
    )


def _build_random_forest(class_weight: Any) -> RandomForestClassifier:
    seed = get_project_config().random_seed
    return RandomForestClassifier(
        n_estimators=300,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=seed,
    )


def _build_hist_gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=get_project_config().random_seed)


def _fit_record_estimator(record: ExperimentRecord, prepared_data: Any) -> Any:
    params = dict(record.hyperparameters)
    if record.model == "XGBoostClassifier":
        params.pop("device", None)
        estimator = build_xgboost_classifier(device="cpu", **params)
        return _fit_with_optional_sample_weight(estimator, prepared_data.X_train, prepared_data.y_train)
    if record.model == "RandomForestClassifier":
        estimator = RandomForestClassifier(**params)
        estimator.fit(prepared_data.X_train, prepared_data.y_train)
        return estimator
    if record.model == "HistGradientBoostingClassifier":
        estimator = HistGradientBoostingClassifier(**params)
        if record.class_weighting == "balanced_sample_weight":
            sample_weight = pd.Series(
                compute_sample_weight(class_weight="balanced", y=prepared_data.y_train),
                index=prepared_data.y_train.index,
            )
            return _fit_with_optional_sample_weight(
                estimator,
                prepared_data.X_train,
                prepared_data.y_train,
                sample_weight=sample_weight,
            )
        estimator.fit(prepared_data.X_train, prepared_data.y_train)
        return estimator
    raise ValueError(f"Unsupported best model reconstruction: {record.model}.")


def _scale_pos_weight_grid(y_train: pd.Series) -> list[float]:
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    return sorted(
        {
            1.0,
            round(0.50 * ratio, 6),
            round(0.75 * ratio, 6),
            round(1.00 * ratio, 6),
            round(1.25 * ratio, 6),
            round(1.50 * ratio, 6),
        }
    )


def _compact_xgb_search_space(y_train: pd.Series) -> dict[str, list[Any]]:
    return {
        "n_estimators": [200, 400, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.25, 0.5],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0.0, 0.5, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "scale_pos_weight": _scale_pos_weight_grid(y_train),
    }


def _sample_xgb_configs(y_train: pd.Series, n_iter: int = 12) -> list[dict[str, Any]]:
    seed = get_project_config().random_seed
    sampled = list(ParameterSampler(_compact_xgb_search_space(y_train), n_iter=n_iter, random_state=seed))
    sampled.sort(
        key=lambda params: (
            params["learning_rate"],
            params["n_estimators"],
            params["max_depth"],
            params["scale_pos_weight"],
        )
    )
    return sampled


def _run_xgb_baseline(prepared_data: Any) -> tuple[ExperimentRecord, Any, pd.DataFrame]:
    seed = get_project_config().random_seed
    base = build_xgboost_classifier(
        scale_pos_weight=_resolve_scale_pos_weight("balanced", prepared_data.y_train),
        device="cpu",
    )
    base = _fit_with_optional_sample_weight(base, prepared_data.X_train, prepared_data.y_train)
    tuning = tune_model("xgboost", base, prepared_data.X_train, prepared_data.y_train)
    metrics, threshold_table, _ = _evaluate_estimator_on_validation(
        estimator=tuning.best_estimator,
        X_validation=prepared_data.X_validation,
        y_validation=prepared_data.y_validation,
    )
    return (
        _make_record(
            experiment="exp0_baseline",
            model="XGBoostClassifier",
            strategy="current_pipeline_tuned_plus_threshold",
            metrics=metrics,
            seed=seed,
            class_weighting=f"scale_pos_weight={tuning.best_params['scale_pos_weight']}",
            hyperparameters=tuning.best_params,
        ),
        tuning.best_estimator,
        threshold_table,
    )


def _run_threshold_only_experiment(
    prepared_data: Any,
    tuned_estimator: Any,
) -> tuple[ExperimentRecord, pd.DataFrame]:
    metrics, threshold_table, _ = _evaluate_estimator_on_validation(
        estimator=tuned_estimator,
        X_validation=prepared_data.X_validation,
        y_validation=prepared_data.y_validation,
    )
    record = _make_record(
        experiment="exp1_threshold_only",
        model="XGBoostClassifier",
        strategy="validation_threshold_search_only",
        metrics=metrics,
        seed=get_project_config().random_seed,
        class_weighting=f"scale_pos_weight={tuned_estimator.get_params()['scale_pos_weight']}",
        hyperparameters=tuned_estimator.get_params(),
    )
    return record, threshold_table


def _run_xgb_scale_pos_weight_sweep(prepared_data: Any) -> tuple[pd.DataFrame, ExperimentRecord]:
    seed = get_project_config().random_seed
    baseline_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    }
    records: list[ExperimentRecord] = []
    for scale_pos_weight in _scale_pos_weight_grid(prepared_data.y_train):
        estimator = build_xgboost_classifier(
            device="cpu",
            scale_pos_weight=scale_pos_weight,
            **baseline_params,
        )
        estimator = _fit_with_optional_sample_weight(estimator, prepared_data.X_train, prepared_data.y_train)
        metrics, _, _ = _evaluate_estimator_on_validation(
            estimator=estimator,
            X_validation=prepared_data.X_validation,
            y_validation=prepared_data.y_validation,
        )
        records.append(
            _make_record(
                experiment="exp2_class_weight",
                model="XGBoostClassifier",
                strategy=f"scale_pos_weight_sweep:{scale_pos_weight}",
                metrics=metrics,
                seed=seed,
                class_weighting=f"scale_pos_weight={scale_pos_weight}",
                hyperparameters=estimator.get_params(),
            )
        )

    frame = pd.DataFrame(asdict(record) for record in records).sort_values(
        ["roc_auc", "recall", "f1_binary"], ascending=[False, False, False]
    )
    valid = frame.loc[(frame["roc_auc"] >= 0.75) & (frame["recall"] >= 0.60)]
    best_frame = valid if not valid.empty else frame
    best_row = best_frame.sort_values(
        ["f1_binary", "recall", "precision"], ascending=[False, False, False]
    ).iloc[0]
    best_record = next(record for record in records if record.strategy == best_row["strategy"])
    return frame.reset_index(drop=True), best_record


def _run_xgb_hyperparameter_search(prepared_data: Any) -> tuple[pd.DataFrame, ExperimentRecord]:
    seed = get_project_config().random_seed
    records: list[ExperimentRecord] = []
    for params in _sample_xgb_configs(prepared_data.y_train):
        estimator = build_xgboost_classifier(device="cpu", **params)
        estimator = _fit_with_optional_sample_weight(estimator, prepared_data.X_train, prepared_data.y_train)
        metrics, _, _ = _evaluate_estimator_on_validation(
            estimator=estimator,
            X_validation=prepared_data.X_validation,
            y_validation=prepared_data.y_validation,
        )
        records.append(
            _make_record(
                experiment="exp3_xgb_tuning",
                model="XGBoostClassifier",
                strategy="randomized_threshold_aware_search",
                metrics=metrics,
                seed=seed,
                class_weighting=f"scale_pos_weight={params['scale_pos_weight']}",
                hyperparameters=estimator.get_params(),
            )
        )

    frame = pd.DataFrame(asdict(record) for record in records)
    valid = frame.loc[(frame["roc_auc"] >= 0.75) & (frame["recall"] >= 0.60)]
    best_frame = valid if not valid.empty else frame
    best_row = best_frame.sort_values(
        ["f1_binary", "recall", "precision", "roc_auc"],
        ascending=[False, False, False, False],
    ).iloc[0]
    best_record = next(
        record
        for record in records
        if record.threshold == best_row["threshold"]
        and record.hyperparameters == best_row["hyperparameters"]
    )
    return frame.sort_values(
        ["f1_binary", "recall", "precision", "roc_auc"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True), best_record


def _best_rf_variant(prepared_data: Any) -> tuple[Any, str]:
    seed = get_project_config().random_seed
    candidates = [
        ("none", _build_random_forest(None)),
        ("balanced", _build_random_forest("balanced")),
        ("balanced_subsample", _build_random_forest("balanced_subsample")),
        ("manual_light", _build_random_forest({0: 1.0, 1: 2.0})),
    ]
    best_metrics: dict[str, Any] | None = None
    best_estimator: Any | None = None
    best_label = "none"
    for label, estimator in candidates:
        estimator.fit(prepared_data.X_train, prepared_data.y_train)
        metrics, _, _ = _evaluate_estimator_on_validation(
            estimator=estimator,
            X_validation=prepared_data.X_validation,
            y_validation=prepared_data.y_validation,
        )
        if best_metrics is None or (
            metrics["f1_binary"],
            metrics["recall_positive"],
            metrics["precision_positive"],
            metrics["roc_auc"],
        ) > (
            best_metrics["f1_binary"],
            best_metrics["recall_positive"],
            best_metrics["precision_positive"],
            best_metrics["roc_auc"],
        ):
            best_metrics = metrics
            best_estimator = estimator
            best_label = label
    assert best_estimator is not None
    return best_estimator, best_label


def _best_hgb_variant(prepared_data: Any) -> tuple[Any, str]:
    base = _build_hist_gradient_boosting()
    base.fit(prepared_data.X_train, prepared_data.y_train)
    base_metrics, _, _ = _evaluate_estimator_on_validation(
        estimator=base,
        X_validation=prepared_data.X_validation,
        y_validation=prepared_data.y_validation,
    )

    weighted = _build_hist_gradient_boosting()
    sample_weight = pd.Series(
        compute_sample_weight(class_weight="balanced", y=prepared_data.y_train),
        index=prepared_data.y_train.index,
    )
    weighted = _fit_with_optional_sample_weight(
        weighted,
        prepared_data.X_train,
        prepared_data.y_train,
        sample_weight=sample_weight,
    )
    weighted_metrics, _, _ = _evaluate_estimator_on_validation(
        estimator=weighted,
        X_validation=prepared_data.X_validation,
        y_validation=prepared_data.y_validation,
    )

    if weighted_metrics["f1_binary"] >= base_metrics["f1_binary"]:
        return weighted, "balanced_sample_weight"
    return base, "none"


def _reevaluate_existing_models(prepared_data: Any, best_xgb_record: ExperimentRecord) -> pd.DataFrame:
    seed = get_project_config().random_seed
    rf_best_base, rf_label = _best_rf_variant(prepared_data)
    rf_tuning = tune_model("random_forest", rf_best_base, prepared_data.X_train, prepared_data.y_train)
    rf_metrics, _, _ = _evaluate_estimator_on_validation(
        estimator=rf_tuning.best_estimator,
        X_validation=prepared_data.X_validation,
        y_validation=prepared_data.y_validation,
    )
    rf_record = _make_record(
        experiment="exp4_model_comparison",
        model="RandomForestClassifier",
        strategy=f"tuned_from_{rf_label}",
        metrics=rf_metrics,
        seed=seed,
        class_weighting=rf_label,
        hyperparameters=rf_tuning.best_params,
    )

    hgb_best_base, hgb_label = _best_hgb_variant(prepared_data)
    sample_weight = None
    if hgb_label == "balanced_sample_weight":
        sample_weight = pd.Series(
            compute_sample_weight(class_weight="balanced", y=prepared_data.y_train),
            index=prepared_data.y_train.index,
        )
    hgb_tuning = tune_model(
        "hist_gradient_boosting",
        hgb_best_base,
        prepared_data.X_train,
        prepared_data.y_train,
        sample_weight=sample_weight,
    )
    hgb_metrics, _, _ = _evaluate_estimator_on_validation(
        estimator=hgb_tuning.best_estimator,
        X_validation=prepared_data.X_validation,
        y_validation=prepared_data.y_validation,
    )
    hgb_record = _make_record(
        experiment="exp4_model_comparison",
        model="HistGradientBoostingClassifier",
        strategy=f"tuned_from_{hgb_label}",
        metrics=hgb_metrics,
        seed=seed,
        class_weighting=hgb_label,
        hyperparameters=hgb_tuning.best_params,
    )

    xgb_row = asdict(best_xgb_record)
    xgb_row["experiment"] = "exp4_model_comparison"
    xgb_row["strategy"] = "best_xgb_after_objective_search"

    comparison = pd.DataFrame([asdict(rf_record), asdict(hgb_record), xgb_row]).sort_values(
        ["status", "f1_binary", "recall", "roc_auc"],
        ascending=[True, False, False, False],
    )
    return comparison.reset_index(drop=True)


def _build_audit_summary() -> pd.DataFrame:
    rows = [
        {
            "arquivo": "src/credit_default/data/load.py",
            "funcao": "normalize_binary_target/load_uci_dataset",
            "responsabilidade": "carregar dataset e normalizar target",
            "comportamento_atual": "normaliza target para 0/1; default=1 permanece positivo",
            "alteracao_necessaria": "nenhuma no contrato; adicionar teste explicito do gate",
        },
        {
            "arquivo": "src/credit_default/data/split.py",
            "funcao": "make_splits",
            "responsabilidade": "split 70/15/15 estratificado deterministico",
            "comportamento_atual": "usa seed 42 e preserva isolamento train/validation/test",
            "alteracao_necessaria": "nenhuma",
        },
        {
            "arquivo": "src/credit_default/features/preprocessing.py",
            "funcao": "fit_preprocessor/transform_features",
            "responsabilidade": "preprocessing reutilizavel fitado so no train",
            "comportamento_atual": "sem leakage",
            "alteracao_necessaria": "nenhuma",
        },
        {
            "arquivo": "src/credit_default/modeling/evaluate.py",
            "funcao": "evaluate_binary_classifier/build_threshold_search_table",
            "responsabilidade": "calculo unico de metricas e threshold search",
            "comportamento_atual": "agora centraliza ROC-AUC, precision, recall, F1 binary/macro/weighted e confusao",
            "alteracao_necessaria": "usar em todos os experimentos",
        },
        {
            "arquivo": "src/credit_default/modeling/tune.py",
            "funcao": "tune_model",
            "responsabilidade": "tuning via CV em train",
            "comportamento_atual": "refit por average_precision",
            "alteracao_necessaria": "comparacao final por modelo+hiperparametros+threshold no workflow experimental",
        },
        {
            "arquivo": "src/credit_default/modeling/train.py",
            "funcao": "run_validation_round_v2/run_modeling_workflow",
            "responsabilidade": "workflow atual de validacao/modelagem",
            "comportamento_atual": "validation-only em v2; workflow legado acessa test no final",
            "alteracao_necessaria": "nao usar workflow legado; usar runner validation-only novo",
        },
    ]
    return pd.DataFrame(rows)


def _plot_precision_recall_curve(y_true: pd.Series, scores: np.ndarray, destination: Path) -> None:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="PR curve")
    plt.axvline(0.60, color="tab:red", linestyle="--", label="Recall gate = 0.60")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def _plot_threshold_curves(threshold_table: pd.DataFrame, destination: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(threshold_table["threshold"], threshold_table["f1_binary"], label="F1 binary")
    plt.plot(threshold_table["threshold"], threshold_table["precision_positive"], label="Precision")
    plt.plot(threshold_table["threshold"], threshold_table["recall_positive"], label="Recall")
    plt.axhline(0.60, color="tab:red", linestyle="--", label="Recall gate = 0.60")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Threshold vs F1 / Precision / Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def _plot_confusion_matrix(metrics: dict[str, Any], destination: Path) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _persist_artifacts(
    *,
    artifact_dir: Path,
    best_record: ExperimentRecord,
    best_metrics: dict[str, Any],
    audit_summary: pd.DataFrame,
    experiment_records: pd.DataFrame,
    threshold_search: pd.DataFrame,
    tuning_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    y_validation: pd.Series,
    best_scores: np.ndarray,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    audit_summary.to_csv(artifact_dir / "audit_summary.csv", index=False)
    threshold_search.to_csv(artifact_dir / "threshold_search.csv", index=False)
    tuning_results.to_csv(artifact_dir / "tuning_results.csv", index=False)
    model_comparison.to_csv(artifact_dir / "model_comparison.csv", index=False)
    experiment_records.to_csv(artifact_dir / "experiment_log.csv", index=False)

    best_configuration = {
        "model": best_record.model,
        "strategy": best_record.strategy,
        "class_weighting": best_record.class_weighting,
        "hyperparameters": best_record.hyperparameters,
        "threshold": best_record.threshold,
        "seed": best_record.seed,
    }
    metrics_summary = {
        "best_result": asdict(best_record),
        "best_metrics": best_metrics,
        "requirements": {
            "roc_auc": REQUIRED_METRIC_GATES["roc_auc"],
            "recall_positive": REQUIRED_METRIC_GATES["recall"],
            "f1_binary": REQUIRED_METRIC_GATES["f1"],
        },
    }
    (artifact_dir / "best_configuration.json").write_text(
        json.dumps(_json_ready(best_configuration), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "metrics_summary.json").write_text(
        json.dumps(_json_ready(metrics_summary), indent=2),
        encoding="utf-8",
    )

    _plot_precision_recall_curve(y_validation, best_scores, artifact_dir / "precision_recall_curve.png")
    _plot_threshold_curves(threshold_search, artifact_dir / "f1_vs_threshold.png")
    _plot_confusion_matrix(best_metrics, artifact_dir / "confusion_matrix.png")


def run_binary_f1_optimization(splits: DatasetSplits) -> OptimizationRunResult:
    """Run the experimental binary-F1 optimization plan on validation only."""

    prepared = prepare_validation_modeling_data(splits)
    audit_summary = _build_audit_summary()

    baseline_record, baseline_estimator, baseline_threshold_search = _run_xgb_baseline(prepared)
    experiment_records = [asdict(baseline_record)]

    threshold_record, threshold_search = _run_threshold_only_experiment(prepared, baseline_estimator)
    experiment_records.append(asdict(threshold_record))

    class_weight_frame, best_weight_record = _run_xgb_scale_pos_weight_sweep(prepared)
    experiment_records.extend(class_weight_frame.to_dict(orient="records"))

    tuning_frame, best_tuned_record = _run_xgb_hyperparameter_search(prepared)
    experiment_records.extend(tuning_frame.to_dict(orient="records"))

    comparison_frame = _reevaluate_existing_models(prepared, best_tuned_record)
    experiment_records.extend(comparison_frame.to_dict(orient="records"))

    experiment_frame = pd.DataFrame(experiment_records)
    best_candidates = experiment_frame.loc[
        (experiment_frame["roc_auc"] >= REQUIRED_METRIC_GATES["roc_auc"])
        & (experiment_frame["recall"] >= REQUIRED_METRIC_GATES["recall"])
    ]
    selection_frame = best_candidates if not best_candidates.empty else experiment_frame
    selected_row = selection_frame.sort_values(
        ["f1_binary", "recall", "precision", "roc_auc"],
        ascending=[False, False, False, False],
    ).iloc[0]
    best_record = ExperimentRecord(
        **{
            **selected_row.to_dict(),
            "hyperparameters": dict(selected_row["hyperparameters"]),
        }
    )

    best_estimator = _fit_record_estimator(best_record, prepared)
    best_metrics, best_threshold_search, best_scores = _evaluate_estimator_on_validation(
        estimator=best_estimator,
        X_validation=prepared.X_validation,
        y_validation=prepared.y_validation,
    )

    model_comparison = comparison_frame.loc[
        :,
        ["experiment", "model", "strategy", "threshold", "precision", "recall", "f1_binary", "roc_auc", "status"],
    ].reset_index(drop=True)

    tuning_results = pd.concat(
        [
            class_weight_frame.assign(experiment_stage="exp2"),
            tuning_frame.assign(experiment_stage="exp3"),
        ],
        ignore_index=True,
    )

    _persist_artifacts(
        artifact_dir=ARTIFACT_DIR,
        best_record=best_record,
        best_metrics=best_metrics,
        audit_summary=audit_summary,
        experiment_records=experiment_frame,
        threshold_search=best_threshold_search,
        tuning_results=tuning_results,
        model_comparison=model_comparison,
        y_validation=prepared.y_validation,
        best_scores=best_scores,
    )

    return OptimizationRunResult(
        audit_summary=audit_summary,
        baseline=baseline_record,
        best_result=best_record,
        best_metrics=best_metrics,
        experiment_records=experiment_frame,
        threshold_search=best_threshold_search,
        model_comparison=model_comparison,
        tuning_results=tuning_results,
        artifact_dir=ARTIFACT_DIR,
    )


def run_binary_f1_optimization_from_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> OptimizationRunResult:
    """Convenience wrapper for scripts/tests."""

    return run_binary_f1_optimization(make_splits(features, target))
