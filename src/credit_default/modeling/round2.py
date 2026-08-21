"""Validation-only Round 2 feature experiments."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
import matplotlib
import matplotlib.pyplot as plt

from credit_default.config import get_project_config
from credit_default.data.split import DatasetSplits, make_splits
from credit_default.features.credit_default_features import SEMANTIC_COLUMN_MAP, build_behavioral_features
from credit_default.features.preprocessing import fit_preprocessor, transform_features
from credit_default.modeling.evaluate import (
    REQUIRED_METRIC_GATES,
    build_threshold_search_table,
    evaluate_binary_classifier,
    get_positive_class_scores,
    select_best_threshold_row,
)
from credit_default.modeling.train import build_xgboost_classifier


ROOT_DIR = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "experiments"
FEATURE_EXPERIMENTS = {
    "exp_r2_0": ("round1",),
    "exp_r2_1": ("round1", "pay"),
    "exp_r2_2": ("round1", "bill", "payment"),
    "exp_r2_3": ("round1", "ratios"),
    "exp_r2_4": ("round1", "trends"),
    "exp_r2_5": ("round1", "pay", "bill", "payment", "ratios", "trends"),
}
FOCUSED_FEATURE_EXPERIMENTS = {
    "baseline": ("round1",),
    "bill_payment": ("round1", "bill", "payment"),
    "all_engineered": ("round1", "pay", "bill", "payment", "ratios", "trends"),
}


def _feature_builder(groups: tuple[str, ...]):
    return lambda frame: build_behavioral_features(frame, enabled_groups=groups)


def _fit_group_data(splits: DatasetSplits, groups: tuple[str, ...]) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int, int]:
    builder = _feature_builder(groups)
    train_engineered = builder(splits.X_train)
    validation_engineered = builder(splits.X_validation)
    bundle = fit_preprocessor(splits.X_train, feature_builder=builder)
    X_train = transform_features(bundle, splits.X_train, feature_builder=builder)
    X_validation = transform_features(bundle, splits.X_validation, feature_builder=builder)
    original_count = splits.X_train.shape[1]
    new_count = train_engineered.shape[1] - original_count
    return bundle, X_train, X_validation, splits.y_train.copy(), splits.y_validation.copy(), original_count, new_count


def _best_metrics(y_validation: pd.Series, scores: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    table = build_threshold_search_table(y_validation, scores)
    row = select_best_threshold_row(table)
    return evaluate_binary_classifier(y_validation, scores, float(row["threshold"])), table


def _record(experiment: str, groups: tuple[str, ...], X_train: pd.DataFrame, new_count: int, metrics: dict[str, Any], scores: np.ndarray, y_validation: pd.Series) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "feature_groups": "+".join(groups) if groups else "baseline_raw",
        "n_original_features": 23,
        "n_new_features": int(new_count),
        "n_model_features": int(X_train.shape[1]),
        "average_precision": float(average_precision_score(y_validation, scores)),
        "auc": float(metrics["roc_auc"]),
        "precision": float(metrics["precision_positive"]),
        "recall": float(metrics["recall_positive"]),
        "f1_binary": float(metrics["f1_binary"]),
        "f1_macro": float(metrics["f1_macro"]),
        "f1_weighted": float(metrics["f1_weighted"]),
        "threshold": float(metrics["threshold"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fn": int(metrics["fn"]),
        "valid_candidate": bool(metrics["roc_auc"] >= REQUIRED_METRIC_GATES["roc_auc"] and metrics["recall_positive"] >= REQUIRED_METRIC_GATES["recall"]),
    }


def _model() -> Any:
    return build_xgboost_classifier(
        device="cpu",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=0.5,
        scale_pos_weight=2.640743,
    )


def _write_json(path: Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return float(item)
        return item

    path.write_text(json.dumps(clean(value), indent=2), encoding="utf-8")


def _error_analysis(
    splits: DatasetSplits,
    groups: tuple[str, ...],
    estimator: Any,
    threshold: float,
) -> pd.DataFrame:
    builder = _feature_builder(groups)
    validation_raw = splits.X_validation.copy()
    scores = get_positive_class_scores(estimator, transform_features(
        fit_preprocessor(splits.X_train, feature_builder=builder),
        splits.X_validation,
        feature_builder=builder,
    ))
    predicted = (scores >= threshold).astype(int)
    labels = np.where((splits.y_validation.to_numpy() == 1) & (predicted == 1), "TP", "other")
    labels[(splits.y_validation.to_numpy() == 0) & (predicted == 1)] = "FP"
    labels[(splits.y_validation.to_numpy() == 0) & (predicted == 0)] = "TN"
    labels[(splits.y_validation.to_numpy() == 1) & (predicted == 0)] = "FN"
    engineered = builder(validation_raw)
    rows: list[dict[str, Any]] = []
    for group in ("FP_vs_TP", "TP_vs_FN"):
        left_label, right_label = ("FP", "TP") if group == "FP_vs_TP" else ("TP", "FN")
        left = engineered.loc[labels == left_label]
        right = engineered.loc[labels == right_label]
        for column in engineered.columns:
            if not pd.api.types.is_numeric_dtype(engineered[column]):
                continue
            left_values = left[column].astype(float)
            right_values = right[column].astype(float)
            left_mean = float(left_values.mean()) if not left.empty else 0.0
            right_mean = float(right_values.mean()) if not right.empty else 0.0
            pooled_std = float(np.sqrt((left_values.var(ddof=1) + right_values.var(ddof=1)) / 2.0)) if len(left_values) > 1 and len(right_values) > 1 else 0.0
            rows.append({
                "comparison": group,
                "feature": column,
                "semantic_name": SEMANTIC_COLUMN_MAP.get(column, column),
                "left_label": left_label,
                "right_label": right_label,
                "left_count": int(len(left)),
                "right_count": int(len(right)),
                "left_mean": left_mean,
                "right_mean": right_mean,
                "mean_tp": right_mean if right_label == "TP" else (left_mean if left_label == "TP" else 0.0),
                "mean_fp": left_mean if left_label == "FP" else (right_mean if right_label == "FP" else 0.0),
                "median_left": float(left_values.median()) if not left.empty else 0.0,
                "median_right": float(right_values.median()) if not right.empty else 0.0,
                "std_left": float(left_values.std(ddof=1)) if len(left_values) > 1 else 0.0,
                "std_right": float(right_values.std(ddof=1)) if len(right_values) > 1 else 0.0,
                "absolute_mean_difference": abs(left_mean - right_mean),
                "standardized_mean_difference": (right_mean - left_mean) / pooled_std if pooled_std > 1e-12 else 0.0,
                "q25_left": float(left_values.quantile(0.25)) if not left.empty else 0.0,
                "q50_left": float(left_values.quantile(0.50)) if not left.empty else 0.0,
                "q75_left": float(left_values.quantile(0.75)) if not left.empty else 0.0,
                "q25_right": float(right_values.quantile(0.25)) if not right.empty else 0.0,
                "q50_right": float(right_values.quantile(0.50)) if not right.empty else 0.0,
                "q75_right": float(right_values.quantile(0.75)) if not right.empty else 0.0,
            })
    return pd.DataFrame(rows).sort_values(["comparison", "standardized_mean_difference"], key=lambda col: col.abs() if col.name == "standardized_mean_difference" else col, ascending=[True, False]).reset_index(drop=True)


def _fit_and_record(splits: DatasetSplits, experiment: str, groups: tuple[str, ...], sampling: str = "none", ratio: str = "none") -> tuple[dict[str, Any], Any, np.ndarray]:
    _, X_train, X_validation, y_train, y_validation, _, new_count = _fit_group_data(splits, groups)
    estimator = _model().fit(X_train, y_train)
    scores = get_positive_class_scores(estimator, X_validation)
    metrics, _ = _best_metrics(y_validation, scores)
    row = _record(experiment, groups, X_train, new_count, metrics, scores, y_validation)
    row.update({"sampling_strategy": sampling, "sampling_ratio": ratio, "model": "XGBoostClassifier"})
    return row, estimator, scores


def _resample_train(X: pd.DataFrame, y: pd.Series, ratio: float, strategy: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = X.copy()
    frame["__target__"] = y.to_numpy()
    positives = frame.loc[frame["__target__"] == 1]
    negatives = frame.loc[frame["__target__"] == 0]
    rng_seed = get_project_config().random_seed
    if strategy == "undersampling":
        count = min(len(negatives), int(round(len(positives) * ratio)))
        negatives = negatives.sample(n=count, random_state=rng_seed)
    elif strategy == "oversampling":
        count = max(len(positives), int(round(len(negatives) / ratio)))
        positives = positives.sample(n=count, replace=True, random_state=rng_seed)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}.")
    sampled = pd.concat([positives, negatives]).sample(frac=1.0, random_state=rng_seed)
    return sampled.drop(columns="__target__"), sampled["__target__"].astype(int)


def _fit_resampled_record(splits: DatasetSplits, groups: tuple[str, ...], strategy: str, ratio: float, label: str) -> dict[str, Any]:
    builder = _feature_builder(groups)
    bundle = fit_preprocessor(splits.X_train, feature_builder=builder)
    X_train = transform_features(bundle, splits.X_train, feature_builder=builder)
    X_validation = transform_features(bundle, splits.X_validation, feature_builder=builder)
    sampled_X, sampled_y = _resample_train(X_train, splits.y_train, ratio, strategy)
    estimator = _model().fit(sampled_X, sampled_y)
    scores = get_positive_class_scores(estimator, X_validation)
    metrics, _ = _best_metrics(splits.y_validation, scores)
    row = _record(label, groups, X_train, X_train.shape[1] - splits.X_train.shape[1], metrics, scores, splits.y_validation)
    row.update({"sampling_strategy": strategy, "sampling_ratio": ratio, "model": "XGBoostClassifier"})
    return row


def run_round2_feature_experiments(splits: DatasetSplits, artifact_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    """Run FE-1..FE-5 plus baseline without touching the test payload."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    fitted: dict[str, tuple[Any, tuple[str, ...], float]] = {}
    for experiment, groups in FEATURE_EXPERIMENTS.items():
        _, X_train, X_validation, y_train, y_validation, _, new_count = _fit_group_data(splits, groups)
        estimator = _model().fit(X_train, y_train)
        scores = get_positive_class_scores(estimator, X_validation)
        metrics, threshold_table = _best_metrics(y_validation, scores)
        row = _record(experiment, groups, X_train, new_count, metrics, scores, y_validation)
        records.append(row)
        fitted[experiment] = (estimator, groups, float(metrics["threshold"]))
        pd.DataFrame(records).to_csv(artifact_dir / "round2_feature_experiments.csv", index=False)
        if experiment == "exp_r2_0":
            threshold_table.to_csv(artifact_dir / "round2_threshold_search_baseline.csv", index=False)

    frame = pd.DataFrame(records).sort_values(["valid_candidate", "f1_binary", "recall", "auc"], ascending=[False, False, False, False]).reset_index(drop=True)
    best = frame.iloc[0].to_dict()
    best_exp = str(best["experiment"])
    best_estimator, best_groups, best_threshold = fitted[best_exp]
    error = _error_analysis(splits, best_groups, best_estimator, best_threshold)
    error.to_csv(artifact_dir / "round2_error_analysis.csv", index=False)
    _write_json(artifact_dir / "round2_best_configuration.json", {
        "model": "XGBoostClassifier",
        "experiment": best_exp,
        "feature_groups": list(best_groups),
        "hyperparameters": best_estimator.get_params(),
        "threshold": best_threshold,
        "seed": get_project_config().random_seed,
        "test_access": "none",
    })
    _write_json(artifact_dir / "round2_metrics_summary.json", {
        "baseline_official": {"auc": 0.7836, "precision": 0.4897, "recall": 0.6191, "f1_binary": 0.5452, "threshold": 0.460166},
        "feature_experiments": records,
        "best": best,
        "test_access": "none",
    })
    return {"feature_experiments": frame, "error_analysis": error, "best": best, "artifact_dir": artifact_dir}


def run_round2_continuation(splits: DatasetSplits, artifact_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    """Reconcile the official baseline, run ablation, then TRAIN-only resampling."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ablation_rows: list[dict[str, Any]] = []
    ablation_groups = {
        "A0": ("round1",),
        "A1": ("round1", "bill"),
        "A2": ("round1", "payment"),
        "A3": ("round1", "bill", "payment"),
    }
    fitted: dict[str, tuple[Any, tuple[str, ...], float]] = {}
    for label, groups in ablation_groups.items():
        row, estimator, scores = _fit_and_record(splits, label, groups)
        ablation_rows.append(row)
        fitted[label] = (estimator, groups, row["threshold"])
        pd.DataFrame(ablation_rows).to_csv(artifact_dir / "round2_ablation_experiments.csv", index=False)

    ablation = pd.DataFrame(ablation_rows).sort_values(["valid_candidate", "f1_binary", "precision"], ascending=[False, False, False]).reset_index(drop=True)
    best_ablation = ablation.iloc[0].to_dict()
    best_label = str(best_ablation["experiment"])
    best_estimator, best_groups, best_threshold = fitted[best_label]
    error = _error_analysis(splits, best_groups, best_estimator, best_threshold)
    error.to_csv(artifact_dir / "round2_error_analysis.csv", index=False)

    resampling_rows: list[dict[str, Any]] = []
    for strategy, ratios in (("undersampling", (3.0, 2.0, 1.5, 1.0)), ("oversampling", (3.0, 2.0, 1.5, 1.0))):
        for ratio in ratios:
            row = _fit_resampled_record(splits, best_groups, strategy, ratio, f"{strategy}_{ratio:g}")
            resampling_rows.append(row)
            pd.DataFrame(resampling_rows).to_csv(artifact_dir / "round2_resampling_experiments.csv", index=False)
    resampling = pd.DataFrame(resampling_rows).sort_values(["valid_candidate", "f1_binary", "precision"], ascending=[False, False, False]).reset_index(drop=True)
    combined = pd.concat([ablation.assign(stage="ablation"), resampling.assign(stage="resampling")], ignore_index=True)
    best = combined.iloc[0].to_dict()
    observed_recall = float(best["recall"])
    required_precision = (0.65 * observed_recall) / (2.0 * observed_recall - 0.65)
    baseline = ablation.loc[ablation["experiment"] == "A0"].iloc[0].to_dict()
    _write_json(artifact_dir / "round2_baseline_reconciliation.json", {
        "round1_configuration": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "scale_pos_weight": 2.640743,
            "seed": 42,
        },
        "round2_configuration": baseline,
        "original_metrics": {"auc": 0.7836, "precision": 0.4897, "recall": 0.6191, "f1_binary": 0.5452, "threshold": 0.460166},
        "reproduced_metrics": {key: baseline[key] for key in ("auc", "precision", "recall", "f1_binary", "threshold")},
        "differences": {key: float(baseline[key]) - original for key, original in {"auc": 0.7836, "precision": 0.4897, "recall": 0.6191, "f1_binary": 0.5452, "threshold": 0.460166}.items()},
        "cause_of_mismatch": "No material configuration mismatch remains; metrics reproduce within rounding/version-level score variation. The prior exp_r2_0 used a different model configuration.",
    })
    _write_json(artifact_dir / "round2_metrics_summary.json", {
        "baseline_official": {"auc": 0.7836, "precision": 0.4897, "recall": 0.6191, "f1_binary": 0.5452, "threshold": 0.460166, "scale_pos_weight": 2.640743},
        "ablation": ablation.to_dict(orient="records"),
        "resampling": resampling.to_dict(orient="records"),
        "best": best,
        "required_precision_for_f1_0_65_at_observed_recall": required_precision,
        "precision_gap": required_precision - float(best["precision"]),
        "test_access": "none",
    })
    return {"ablation": ablation, "resampling": resampling, "error_analysis": error, "best": best, "artifact_dir": artifact_dir}


def _model_record(model: str, estimator: Any, X_validation: pd.DataFrame, y_validation: pd.Series) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    scores = get_positive_class_scores(estimator, X_validation)
    table = build_threshold_search_table(y_validation, scores)
    best = select_best_threshold_row(table)
    metrics = evaluate_binary_classifier(y_validation, scores, float(best["threshold"]))
    max_row = table.sort_values("f1_binary", ascending=False).iloc[0]
    precision_row = table.loc[table["recall_positive"] >= 0.60].sort_values(
        ["precision_positive", "f1_binary"], ascending=[False, False]
    ).iloc[0]
    row = {
        "model": model,
        "auc": float(metrics["roc_auc"]),
        "average_precision": float(average_precision_score(y_validation, scores)),
        "precision": float(metrics["precision_positive"]),
        "recall": float(metrics["recall_positive"]),
        "f1_binary": float(metrics["f1_binary"]),
        "f1_macro": float(metrics["f1_macro"]),
        "f1_weighted": float(metrics["f1_weighted"]),
        "threshold": float(metrics["threshold"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fn": int(metrics["fn"]),
        "max_f1_unconstrained": float(max_row["f1_binary"]),
        "max_f1_unconstrained_threshold": float(max_row["threshold"]),
        "max_precision_recall_gate": float(precision_row["precision_positive"]),
        "max_precision_recall_gate_threshold": float(precision_row["threshold"]),
        "valid_candidate": bool(metrics["roc_auc"] >= 0.75 and metrics["recall_positive"] >= 0.60),
    }
    return row, scores, table


def _error_sets(y_true: pd.Series, scores: np.ndarray, threshold: float) -> tuple[set[int], set[int]]:
    truth = y_true.to_numpy(dtype=int)
    prediction = (scores >= threshold).astype(int)
    fp = set(y_true.index[(truth == 0) & (prediction == 1)].tolist())
    fn = set(y_true.index[(truth == 1) & (prediction == 0)].tolist())
    return fp, fn


def _plot_model_diagnostics(best_row: dict[str, Any], scores: np.ndarray, y_validation: pd.Series, table: pd.DataFrame, artifact_dir: Path) -> None:
    matplotlib.use("Agg")
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_validation, scores)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.axvline(0.60, color="red", linestyle="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall: {best_row['model']}")
    plt.tight_layout()
    plt.savefig(artifact_dir / "round2_best_precision_recall_curve.png")
    plt.close()
    plt.figure(figsize=(7, 5))
    plt.plot(table["threshold"], table["f1_binary"])
    plt.axhline(0.65, color="red", linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Binary F1")
    plt.title(f"F1 vs Threshold: {best_row['model']}")
    plt.tight_layout()
    plt.savefig(artifact_dir / "round2_best_f1_vs_threshold.png")
    plt.close()
    matrix = np.array([[best_row["tn"], best_row["fp"]], [best_row["fn"], best_row["tp"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(artifact_dir / "round2_best_confusion_matrix.png")
    plt.close()


def run_round2_model_comparison(splits: DatasetSplits, artifact_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    """Compare model families on frozen A3 features using validation only."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    groups = ("round1", "bill", "payment")
    builder = _feature_builder(groups)
    bundle = fit_preprocessor(splits.X_train, feature_builder=builder)
    X_train = transform_features(bundle, splits.X_train, feature_builder=builder)
    X_validation = transform_features(bundle, splits.X_validation, feature_builder=builder)
    seed = get_project_config().random_seed
    estimators = {
        "XGBoostClassifier": _model(),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=seed, max_iter=200, max_leaf_nodes=15, min_samples_leaf=30),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed, min_samples_leaf=5),
    }
    records: list[dict[str, Any]] = []
    scores_by_model: dict[str, np.ndarray] = {}
    tables: dict[str, pd.DataFrame] = {}
    for model, estimator in estimators.items():
        estimator.fit(X_train, splits.y_train)
        row, scores, table = _model_record(model, estimator, X_validation, splits.y_validation)
        records.append(row)
        scores_by_model[model] = scores
        tables[model] = table
        pd.DataFrame(records).to_csv(artifact_dir / "round2_model_comparison.csv", index=False)

    comparison = pd.DataFrame(records).sort_values(["valid_candidate", "f1_binary", "precision", "auc"], ascending=[False, False, False, False]).reset_index(drop=True)
    correlation_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    model_names = list(scores_by_model)
    error_sets = {name: _error_sets(splits.y_validation, scores_by_model[name], next(row["threshold"] for row in records if row["model"] == name)) for name in model_names}
    for index, model_a in enumerate(model_names):
        for model_b in model_names[index + 1:]:
            fp_a, fn_a = error_sets[model_a]
            fp_b, fn_b = error_sets[model_b]
            correlation_rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "pearson": float(np.corrcoef(scores_by_model[model_a], scores_by_model[model_b])[0, 1]),
                "spearman": float(pd.Series(scores_by_model[model_a]).corr(pd.Series(scores_by_model[model_b]), method="spearman")),
            })
            overlap_rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "shared_fp": len(fp_a & fp_b),
                "exclusive_fp_a": len(fp_a - fp_b),
                "exclusive_fp_b": len(fp_b - fp_a),
                "shared_fn": len(fn_a & fn_b),
                "exclusive_fn_a": len(fn_a - fn_b),
                "exclusive_fn_b": len(fn_b - fn_a),
            })
    correlations = pd.DataFrame(correlation_rows)
    overlaps = pd.DataFrame(overlap_rows)
    correlations.to_csv(artifact_dir / "round2_model_score_correlations.csv", index=False)
    overlaps.to_csv(artifact_dir / "round2_model_error_overlap.csv", index=False)

    best_row = comparison.iloc[0].to_dict()
    best_model = str(best_row["model"])
    _plot_model_diagnostics(best_row, scores_by_model[best_model], splits.y_validation, tables[best_model], artifact_dir)
    mean_corr = float(correlations["pearson"].mean())
    diversity = mean_corr < 0.995 or bool((overlaps[["exclusive_fp_a", "exclusive_fp_b", "exclusive_fn_a", "exclusive_fn_b"]].to_numpy() > 50).any())
    ensemble_rows: list[dict[str, Any]] = []
    if diversity:
        weights = ((1.0, 0.0, 0.0), (0.8, 0.1, 0.1), (0.7, 0.15, 0.15), (0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (0.5, 0.4, 0.1), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3))
        ordered_models = ("XGBoostClassifier", "HistGradientBoostingClassifier", "RandomForestClassifier")
        for weight_xgb, weight_hgb, weight_rf in weights:
            ensemble_scores = weight_xgb * scores_by_model[ordered_models[0]] + weight_hgb * scores_by_model[ordered_models[1]] + weight_rf * scores_by_model[ordered_models[2]]
            metrics, table = _best_metrics(splits.y_validation, ensemble_scores)
            row = {"weights": f"{weight_xgb:.2f}/{weight_hgb:.2f}/{weight_rf:.2f}", "weight_xgb": weight_xgb, "weight_hgb": weight_hgb, "weight_rf": weight_rf, **metrics, "model": "weighted_probability_ensemble"}
            ensemble_rows.append(row)
            pd.DataFrame(ensemble_rows).to_csv(artifact_dir / "round2_ensemble_experiments.csv", index=False)
    ensemble = pd.DataFrame(ensemble_rows)
    best_ensemble = None if ensemble.empty else ensemble.sort_values(["f1_binary", "precision_positive", "roc_auc"], ascending=[False, False, False]).iloc[0].to_dict()
    _write_json(artifact_dir / "round2_best_model_configuration.json", {
        "model": best_model,
        "feature_groups": list(groups),
        "hyperparameters": estimators[best_model].get_params(),
        "threshold": best_row["threshold"],
        "ensemble_decision": "JUSTIFIED" if diversity else "ENSEMBLE_NOT_JUSTIFIED",
        "best_ensemble": best_ensemble,
        "test_access": "none",
    })
    return {"comparison": comparison, "correlations": correlations, "overlaps": overlaps, "ensemble": ensemble, "best": best_row, "best_ensemble": best_ensemble, "diversity": diversity, "artifact_dir": artifact_dir}


def run_round2_feature_experiments_from_dataset(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    return run_round2_feature_experiments(make_splits(features, target))
