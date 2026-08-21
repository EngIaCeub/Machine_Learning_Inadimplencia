"""Evaluation helpers with explicit A1 macro-F1 gate semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


REQUIRED_METRIC_GATES = {
    "roc_auc": 0.75,
    "f1": 0.65,
    "recall": 0.60,
}
OFFICIAL_A1_F1_METRIC = "f1_macro"


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    split_name: str
    threshold: float
    tuned: bool
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    roc_curve: dict[str, list[float]]
    hyperparameters: dict[str, Any]


@dataclass(frozen=True)
class ValidationThresholdSummary:
    average_precision: float
    f1_at_point_five: float
    max_f1: float
    max_f1_threshold: float
    recall_at_max_f1: float
    precision_at_max_f1: float
    constrained_max_f1: float
    constrained_max_f1_threshold: float
    constrained_recall: float
    constrained_precision: float
    best_precision_at_recall_gate: float
    threshold_at_best_precision_recall_gate: float
    threshold_table: pd.DataFrame


def passes_academic_gates(metrics: dict[str, float]) -> bool:
    official_f1 = metrics.get("official_a1_f1", metrics.get("f1_macro", metrics.get("f1", float("-inf"))))
    return (
        metrics.get("roc_auc", float("-inf")) >= REQUIRED_METRIC_GATES["roc_auc"]
        and metrics.get("recall", metrics.get("recall_positive", float("-inf"))) >= REQUIRED_METRIC_GATES["recall"]
        and official_f1 >= REQUIRED_METRIC_GATES["f1"]
    )


def get_positive_class_scores(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores for thresholding and ROC-AUC."""

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    if hasattr(estimator, "decision_function"):
        decision_scores = estimator.decision_function(features)
        decision_scores = np.asarray(decision_scores, dtype=float)
        minimum = decision_scores.min()
        maximum = decision_scores.max()
        if maximum == minimum:
            return np.full_like(decision_scores, 0.5, dtype=float)
        return (decision_scores - minimum) / (maximum - minimum)
    raise TypeError("Estimator must expose predict_proba or decision_function.")


def evaluate_binary_classifier(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate a binary classifier with explicit positive-class semantics."""

    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).astype(int).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    result: dict[str, float | int] = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "precision_positive": float(
            precision_score(truth, predictions, average="binary", pos_label=1, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(truth, predictions, average="binary", pos_label=1, zero_division=0)
        ),
        "f1_binary": float(
            f1_score(truth, predictions, average="binary", pos_label=1, zero_division=0)
        ),
        "f1_binary_default_1": float(
            f1_score(truth, predictions, average="binary", pos_label=1, zero_division=0)
        ),
        "f1_class_0": float(f1_score(truth, predictions, labels=[0], average="macro", zero_division=0)),
        "f1_macro": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(truth, predictions, average="micro", zero_division=0)),
        "official_a1_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "official_f1_metric": OFFICIAL_A1_F1_METRIC,
        "accuracy": float(accuracy_score(truth, predictions)),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "target_positive_prevalence": float(truth.mean()),
        "predicted_positive_prevalence": float(predictions.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return result


def calculate_classification_metrics(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], list[list[int]], dict[str, list[float]]]:
    """Compute required S03 metrics for a fixed threshold."""

    binary_metrics = evaluate_binary_classifier(y_true, scores, threshold=threshold)
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    metrics = {
        "roc_auc": float(binary_metrics["roc_auc"]),
        "f1": float(binary_metrics["f1_binary"]),
        "recall": float(binary_metrics["recall_positive"]),
        "precision": float(binary_metrics["precision_positive"]),
        "accuracy": float(binary_metrics["accuracy"]),
    }
    matrix = [
        [int(binary_metrics["tn"]), int(binary_metrics["fp"])],
        [int(binary_metrics["fn"]), int(binary_metrics["tp"])],
    ]
    curve = {
        "fpr": fpr.astype(float).tolist(),
        "tpr": tpr.astype(float).tolist(),
        "thresholds": roc_thresholds.astype(float).tolist(),
    }
    return metrics, matrix, curve


def build_threshold_search_table(
    y_true: pd.Series,
    scores: np.ndarray,
    min_recall: float = REQUIRED_METRIC_GATES["recall"],
    min_roc_auc: float = REQUIRED_METRIC_GATES["roc_auc"],
) -> pd.DataFrame:
    """Build a full validation threshold table using one metric implementation."""

    _, _, thresholds = precision_recall_curve(y_true, scores)
    candidate_thresholds = np.unique(
        np.concatenate(([0.5], np.round(np.asarray(thresholds, dtype=float), 6)))
    )
    rows: list[dict[str, float | int | bool]] = []
    for threshold in candidate_thresholds:
        metrics = evaluate_binary_classifier(y_true, scores, threshold=float(threshold))
        row = {
            **metrics,
            "f1": metrics["f1_binary"],
            "precision": metrics["precision_positive"],
            "recall": metrics["recall_positive"],
            "passes_recall_gate": float(metrics["recall_positive"]) >= min_recall,
            "passes_auc_gate": float(metrics["roc_auc"]) >= min_roc_auc,
            "valid_candidate": (
                float(metrics["recall_positive"]) >= min_recall
                and float(metrics["roc_auc"]) >= min_roc_auc
            ),
            "threshold_distance_to_half": abs(float(threshold) - 0.5),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_best_threshold_row(
    threshold_table: pd.DataFrame,
    min_recall: float = REQUIRED_METRIC_GATES["recall"],
    min_roc_auc: float = REQUIRED_METRIC_GATES["roc_auc"],
) -> pd.Series:
    """Select the best threshold row under the binary-F1 objective."""

    valid = threshold_table.loc[
        (threshold_table["recall_positive"] >= min_recall)
        & (threshold_table["roc_auc"] >= min_roc_auc)
    ]
    if not valid.empty:
        return valid.sort_values(
            ["f1_binary", "recall_positive", "precision_positive", "threshold_distance_to_half"],
            ascending=[False, False, False, True],
        ).iloc[0]

    recall_only = threshold_table.loc[threshold_table["recall_positive"] >= min_recall]
    if not recall_only.empty:
        return recall_only.sort_values(
            ["f1_binary", "recall_positive", "precision_positive", "threshold_distance_to_half"],
            ascending=[False, False, False, True],
        ).iloc[0]

    return threshold_table.sort_values(
        ["recall_positive", "f1_binary", "precision_positive", "threshold_distance_to_half"],
        ascending=[False, False, False, True],
    ).iloc[0]


def evaluate_estimator(
    model_name: str,
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.5,
    split_name: str = "validation",
    tuned: bool = False,
) -> EvaluationResult:
    """Evaluate a fitted estimator on validation or test."""

    scores = get_positive_class_scores(estimator, features)
    metrics, matrix, curve = calculate_classification_metrics(target, scores, threshold=threshold)
    return EvaluationResult(
        model_name=model_name,
        split_name=split_name,
        threshold=threshold,
        tuned=tuned,
        metrics=metrics,
        confusion_matrix=matrix,
        roc_curve=curve,
        hyperparameters=estimator.get_params(deep=False),
    )


def tune_decision_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    min_recall: float = REQUIRED_METRIC_GATES["recall"],
) -> tuple[float, pd.DataFrame]:
    """Select a validation threshold with explicit recall guarding."""

    table = build_threshold_search_table(y_true, scores, min_recall=min_recall)
    best_row = select_best_threshold_row(table, min_recall=min_recall)
    return float(best_row["threshold"]), table


def summarize_precision_recall_tradeoff(
    y_true: pd.Series,
    scores: np.ndarray,
    min_recall: float = REQUIRED_METRIC_GATES["recall"],
) -> ValidationThresholdSummary:
    """Summarize validation-only PR behavior for candidate comparison."""

    table = build_threshold_search_table(y_true, scores, min_recall=min_recall)
    max_f1_row = table.sort_values(
        ["f1_binary", "precision_positive", "recall_positive", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    recall_gate_rows = table.loc[table["passes_recall_gate"]]
    if not recall_gate_rows.empty:
        constrained_best_f1_row = recall_gate_rows.sort_values(
            ["f1_binary", "precision_positive", "threshold_distance_to_half"],
            ascending=[False, False, True],
        ).iloc[0]
        best_precision_row = recall_gate_rows.sort_values(
            ["precision_positive", "f1_binary", "threshold_distance_to_half"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        constrained_best_f1_row = max_f1_row
        best_precision_row = max_f1_row

    metrics_at_point_five = evaluate_binary_classifier(y_true, scores, threshold=0.5)
    return ValidationThresholdSummary(
        average_precision=float(average_precision_score(y_true, scores)),
        f1_at_point_five=float(metrics_at_point_five["f1_binary"]),
        max_f1=float(max_f1_row["f1_binary"]),
        max_f1_threshold=float(max_f1_row["threshold"]),
        recall_at_max_f1=float(max_f1_row["recall_positive"]),
        precision_at_max_f1=float(max_f1_row["precision_positive"]),
        constrained_max_f1=float(constrained_best_f1_row["f1_binary"]),
        constrained_max_f1_threshold=float(constrained_best_f1_row["threshold"]),
        constrained_recall=float(constrained_best_f1_row["recall_positive"]),
        constrained_precision=float(constrained_best_f1_row["precision_positive"]),
        best_precision_at_recall_gate=float(best_precision_row["precision_positive"]),
        threshold_at_best_precision_recall_gate=float(best_precision_row["threshold"]),
        threshold_table=table,
    )


def comparison_frame(results: list[EvaluationResult]) -> pd.DataFrame:
    """Build an objective comparison table for candidate selection."""

    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "model_name": result.model_name,
                "variant": "tuned" if result.tuned else "base",
                "split_name": result.split_name,
                "threshold": result.threshold,
                "roc_auc": result.metrics["roc_auc"],
                "f1": result.metrics["f1"],
                "recall": result.metrics["recall"],
                "precision": result.metrics["precision"],
                "accuracy": result.metrics["accuracy"],
                "passes_gates": passes_academic_gates(result.metrics),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["passes_gates", "f1", "roc_auc", "recall", "precision", "accuracy"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def select_best_result(results: list[EvaluationResult]) -> EvaluationResult:
    """Pick the current champion candidate using transparent metric ordering."""

    ordered = sorted(
        results,
        key=lambda result: (
            passes_academic_gates(result.metrics),
            result.metrics["f1"],
            result.metrics["roc_auc"],
            result.metrics["recall"],
            result.metrics["precision"],
            result.metrics["accuracy"],
        ),
        reverse=True,
    )
    return ordered[0]
