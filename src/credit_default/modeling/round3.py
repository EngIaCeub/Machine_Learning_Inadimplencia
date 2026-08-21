"""Round 3 validation-only hard-negative and precision experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from credit_default.config import get_project_config
from credit_default.data.split import DatasetSplits, make_splits
from credit_default.features.preprocessing import fit_preprocessor, transform_features
from credit_default.modeling.evaluate import (
    build_threshold_search_table,
    evaluate_binary_classifier,
    get_positive_class_scores,
    select_best_threshold_row,
)
from credit_default.modeling.round2 import ARTIFACT_DIR, _feature_builder, _model, _write_json


ROUND3_ARTIFACT_DIR = ARTIFACT_DIR
GROUPS = ("round1", "bill", "payment")


def _prepare(train: pd.DataFrame, target: pd.Series, validation: pd.DataFrame) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    builder = _feature_builder(GROUPS)
    bundle = fit_preprocessor(train, feature_builder=builder)
    return bundle, transform_features(bundle, train, feature_builder=builder), transform_features(bundle, validation, feature_builder=builder)


def _select_metrics(y_true: pd.Series, scores: np.ndarray, objective: str = "f1") -> tuple[dict[str, Any], pd.DataFrame]:
    table = build_threshold_search_table(y_true, scores)
    valid = table.loc[table["recall_positive"] >= 0.60]
    if valid.empty:
        row = select_best_threshold_row(table)
    else:
        columns = ["precision_positive", "f1_binary", "threshold"] if objective == "precision" else ["f1_binary", "precision_positive", "threshold"]
        row = valid.sort_values(columns, ascending=[False, False, False]).iloc[0]
    return evaluate_binary_classifier(y_true, scores, float(row["threshold"])), table


def _record(experiment: str, phase: str, strategy: str, model: str, metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "experiment_id": experiment,
        "phase": phase,
        "strategy": strategy,
        "model": model,
        "feature_set": "ROUND1+BILL+PAYMENT",
        "threshold": float(metrics["threshold"]),
        "auc": float(metrics["roc_auc"]),
        "precision": float(metrics["precision_positive"]),
        "recall": float(metrics["recall_positive"]),
        "f1_binary": float(metrics["f1_binary"]),
        "f1_macro": float(metrics["f1_macro"]),
        "f1_weighted": float(metrics["f1_weighted"]),
        "tp": int(metrics["tp"]), "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]), "fn": int(metrics["fn"]),
        "seed": get_project_config().random_seed,
        **extra,
    }


def _oof_predictions(X: pd.DataFrame, y: pd.Series, artifact_dir: Path) -> pd.DataFrame:
    existing_path = artifact_dir / "round3_oof_predictions.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        if len(existing) == len(X) and existing["row_index"].nunique() == len(X):
            return existing
    oof = np.full(len(X), np.nan, dtype=float)
    fold_ids = np.full(len(X), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_pos, holdout_pos) in enumerate(splitter.split(X, y)):
        raw_train = X.iloc[train_pos]
        raw_holdout = X.iloc[holdout_pos]
        y_train = y.iloc[train_pos]
        _, X_fold, X_holdout = _prepare(raw_train, y_train, raw_holdout)
        estimator = _model().fit(X_fold, y_train)
        oof[holdout_pos] = get_positive_class_scores(estimator, X_holdout)
        fold_ids[holdout_pos] = fold
    if np.isnan(oof).any() or (fold_ids < 0).any():
        raise AssertionError("Every training row must receive exactly one OOF prediction.")
    result = pd.DataFrame({"row_index": X.index, "y_true": y.to_numpy(), "oof_score": oof, "fold": fold_ids})
    result["is_hard_negative"] = False
    result.to_csv(artifact_dir / "round3_oof_predictions.csv", index=False)
    return result


def _hard_negative_masks(oof: pd.DataFrame) -> dict[str, pd.Series]:
    negatives = oof.loc[oof["y_true"] == 0, "oof_score"]
    masks: dict[str, pd.Series] = {}
    for quantile in (0.05, 0.10, 0.20):
        cutoff = float(negatives.quantile(1.0 - quantile))
        masks[f"top_{int(quantile * 100)}pct"] = (oof["y_true"] == 0) & (oof["oof_score"] >= cutoff)
    return masks


def _fit_weighted(train_X: pd.DataFrame, y: pd.Series, val_X: pd.DataFrame, mask: pd.Series, weight: float, scale: float) -> tuple[dict[str, Any], np.ndarray]:
    estimator = _model()
    estimator.set_params(scale_pos_weight=scale)
    sample_weight = np.ones(len(y), dtype=float)
    sample_weight[mask.to_numpy()] = weight
    estimator.fit(train_X, y, sample_weight=sample_weight)
    scores = get_positive_class_scores(estimator, val_X)
    metrics, _ = _select_metrics(pd.Series(val_X.index.map(lambda idx: 0)) if False else _CURRENT_Y_VALIDATION, scores, objective="precision")
    return metrics, scores


_CURRENT_Y_VALIDATION: pd.Series


def _hard_negative_phase(splits: DatasetSplits, artifact_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    global _CURRENT_Y_VALIDATION
    _, X_train, X_validation = _prepare(splits.X_train, splits.y_train, splits.X_validation)
    _CURRENT_Y_VALIDATION = splits.y_validation
    baseline_estimator = _model().fit(X_train, splits.y_train)
    baseline_scores = get_positive_class_scores(baseline_estimator, X_validation)
    baseline_metrics, baseline_table = _select_metrics(splits.y_validation, baseline_scores, objective="f1")
    baseline = _record("round3_baseline", "baseline", "A3_xgboost", "XGBoostClassifier", baseline_metrics, hard_negative_definition=None, hard_negative_weight=1.0, scale_pos_weight=2.640743)
    _write_json(artifact_dir / "round3_baseline.json", baseline)
    oof = _oof_predictions(splits.X_train, splits.y_train, artifact_dir)
    masks = _hard_negative_masks(oof)
    analysis_rows = []
    for name, mask in masks.items():
        values = oof.loc[mask, "oof_score"]
        negatives = oof.loc[oof["y_true"] == 0, "oof_score"]
        analysis_rows.append({"definition": name, "score_cutoff": float(values.min()), "hard_negative_count": int(mask.sum()), "negative_percentage": float(mask.sum() / len(negatives)), "mean_score": float(values.mean()), "median_score": float(values.median())})
        oof.loc[mask, "is_hard_negative"] = True
    analysis = pd.DataFrame(analysis_rows)
    analysis.to_csv(artifact_dir / "round3_hard_negative_analysis.csv", index=False)
    records = [baseline]
    for name, mask in masks.items():
        controls = [(2.0, 2.640743), (3.0, 2.640743)]
        if name == "top_10pct":
            controls.append((2.0, 1.0))
        for weight, scale in controls:
            sample_weight = np.ones(len(splits.y_train), dtype=float)
            sample_weight[mask.to_numpy()] = weight
            estimator = _model()
            estimator.set_params(scale_pos_weight=scale)
            estimator.fit(X_train, splits.y_train, sample_weight=sample_weight)
            scores = get_positive_class_scores(estimator, X_validation)
            metrics, _ = _select_metrics(splits.y_validation, scores, objective="precision")
            records.append(_record(f"hn_{name}_{weight:g}_spw_{scale:g}", "3A", "hard_negative_weighting", "XGBoostClassifier", metrics, hard_negative_definition=name, hard_negative_weight=weight, scale_pos_weight=scale))
            pd.DataFrame(records).to_csv(artifact_dir / "round3_hard_negative_experiments.csv", index=False)
    return pd.DataFrame(records), oof, analysis, baseline, baseline_table, X_train


def _cascade_phase(splits: DatasetSplits, oof: pd.DataFrame, X_train: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    builder = _feature_builder(GROUPS)
    bundle = fit_preprocessor(splits.X_train, feature_builder=builder)
    X_validation = transform_features(bundle, splits.X_validation, feature_builder=builder)
    stage1 = _model().fit(X_train, splits.y_train)
    val_stage1 = get_positive_class_scores(stage1, X_validation)
    rows = []
    for central in (0.20, 0.30, 0.40):
        low = float(oof["oof_score"].quantile((1 - central) / 2))
        high = float(oof["oof_score"].quantile(1 - (1 - central) / 2))
        mask = (oof["oof_score"] >= low) & (oof["oof_score"] <= high)
        specialist = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        specialist.fit(X_train.loc[mask.to_numpy()].assign(stage1_score=oof.loc[mask, "oof_score"].to_numpy()), splits.y_train.loc[mask.to_numpy()])
        val_specialist_X = X_validation.assign(stage1_score=val_stage1)
        specialist_scores = specialist.predict_proba(val_specialist_X)[:, 1]
        final_scores = val_stage1.copy()
        val_mask = (val_stage1 >= low) & (val_stage1 <= high)
        final_scores[val_mask] = specialist_scores[val_mask]
        metrics, _ = _select_metrics(splits.y_validation, final_scores, objective="precision")
        rows.append(_record(f"cascade_central_{int(central * 100)}", "3B", "cascade", "XGBoost+LogisticRegression", metrics, ambiguous_lower=low, ambiguous_upper=high, stage2_model="LogisticRegression", ambiguous_train_count=int(mask.sum()), ambiguous_positive_rate=float(splits.y_train.loc[mask.to_numpy()].mean())))
        pd.DataFrame(rows).to_csv(artifact_dir / "round3_cascade_experiments.csv", index=False)
    return pd.DataFrame(rows)


def run_round3(splits: DatasetSplits, artifact_dir: Path = ROUND3_ARTIFACT_DIR) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    hn_records, oof, hn_analysis, baseline, baseline_table, X_train = _hard_negative_phase(splits, artifact_dir)
    cascade = _cascade_phase(splits, oof, X_train, artifact_dir)
    all_records = pd.concat([hn_records, cascade], ignore_index=True)
    valid = all_records.loc[(all_records["auc"] >= 0.75) & (all_records["recall"] >= 0.60)]
    best = (valid if not valid.empty else all_records).sort_values(["f1_binary", "precision"], ascending=[False, False]).iloc[0].to_dict()
    _write_json(artifact_dir / "round3_metrics_summary.json", {"baseline": baseline, "hard_negative": hn_records.to_dict(orient="records"), "cascade": cascade.to_dict(orient="records"), "best": best, "catboost": {"executed": False, "reason": "catboost dependency is not installed"}, "focal": {"executed": False, "reason": "no safe native custom objective in current dependency set"}, "stacking": "NOT_JUSTIFIED", "test_access": "none"})
    _write_json(artifact_dir / "round3_best_configuration.json", {"configuration": best, "test_access": "none"})
    return {"baseline": baseline, "hard_negative": hn_records, "hard_negative_analysis": hn_analysis, "cascade": cascade, "best": best, "oof": oof, "artifact_dir": artifact_dir}


def run_round3_from_dataset(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    return run_round3(make_splits(features, target))


def run_round3_diagnostics(splits: DatasetSplits, artifact_dir: Path = ROUND3_ARTIFACT_DIR) -> dict[str, Any]:
    """Run OOF-derived segmented thresholds and TRAIN separability diagnostics."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    oof_path = artifact_dir / "round3_oof_predictions.csv"
    oof = pd.read_csv(oof_path) if oof_path.exists() else _oof_predictions(splits.X_train, splits.y_train, artifact_dir)
    _, X_train, X_validation = _prepare(splits.X_train, splits.y_train, splits.X_validation)
    stage1 = _model().fit(X_train, splits.y_train)
    validation_scores = get_positive_class_scores(stage1, X_validation)

    pay_cols = ["X6", "X7", "X8", "X9", "X10", "X11"]
    train_delay_count = splits.X_train[pay_cols].gt(0).sum(axis=1)
    validation_delay_count = splits.X_validation[pay_cols].gt(0).sum(axis=1)
    def segment(values: pd.Series) -> pd.Series:
        return pd.cut(values, bins=[-1, 0, 2, np.inf], labels=["no_delay", "light_delay", "recurrent_delay"])
    train_segments = segment(train_delay_count).astype(str)
    validation_segments = segment(validation_delay_count).astype(str)
    thresholds: dict[str, float] = {}
    for name in train_segments.unique():
        rows = oof.loc[train_segments.to_numpy() == name]
        table = build_threshold_search_table(rows["y_true"], rows["oof_score"])
        thresholds[name] = float(select_best_threshold_row(table)["threshold"])
    segmented_predictions = np.zeros(len(validation_scores), dtype=int)
    for name, threshold in thresholds.items():
        mask = validation_segments.to_numpy() == name
        segmented_predictions[mask] = (validation_scores[mask] >= threshold).astype(int)
    segmented_scores = validation_scores.copy()
    metrics = evaluate_binary_classifier(splits.y_validation, segmented_scores, float(np.median(list(thresholds.values()))))
    # Use the segment decisions directly for the global confusion metrics.
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    tn, fp, fn, tp = confusion_matrix(splits.y_validation, segmented_predictions, labels=[0, 1]).ravel()
    metrics.update({"threshold": "segment_specific", "precision_positive": float(precision_score(splits.y_validation, segmented_predictions, zero_division=0)), "recall_positive": float(recall_score(splits.y_validation, segmented_predictions, zero_division=0)), "f1_binary": float(f1_score(splits.y_validation, segmented_predictions, zero_division=0)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    segmented_row = {"experiment_id": "segmented_delay_thresholds", "phase": "3D", "strategy": "OOF_delay_segments", "model": "XGBoostClassifier", "feature_set": "ROUND1+BILL+PAYMENT", "thresholds": json.dumps(thresholds), **metrics, "test_access": "none"}
    pd.DataFrame([segmented_row]).to_csv(artifact_dir / "round3_segmented_thresholds.csv", index=False)

    raw_numeric = splits.X_train.select_dtypes(include=["number"]).copy()
    duplicate_groups = raw_numeric.groupby(list(raw_numeric.columns), dropna=False).size()
    duplicate_indices = duplicate_groups[duplicate_groups > 1]
    duplicate_conflicts = 0
    for key in duplicate_indices.index:
        mask = (raw_numeric == pd.Series(key, index=raw_numeric.columns)).all(axis=1)
        if splits.y_train.loc[mask].nunique() > 1:
            duplicate_conflicts += 1
    scaled = (X_train - X_train.mean()) / X_train.std().replace(0, 1)
    neighbors = NearestNeighbors(n_neighbors=2).fit(scaled)
    near_indices = neighbors.kneighbors(return_distance=False)[:, 1]
    near_opposite_rate = float((splits.y_train.to_numpy() != splits.y_train.to_numpy()[near_indices]).mean())
    oof_neg = oof.loc[oof["y_true"] == 0, "oof_score"]
    oof_pos = oof.loc[oof["y_true"] == 1, "oof_score"]
    overlap = {"negative_percentiles": {str(q): float(oof_neg.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}, "positive_percentiles": {str(q): float(oof_pos.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}, "negative_above_positive_median": float((oof_neg >= oof_pos.median()).mean()), "positive_below_negative_median": float((oof_pos <= oof_neg.median()).mean())}
    _write_json(artifact_dir / "round3_duplicate_analysis.json", {"exact_duplicate_feature_groups": int(len(duplicate_indices)), "conflicting_label_duplicate_groups": int(duplicate_conflicts), "test_access": "none"})
    _write_json(artifact_dir / "round3_score_overlap.json", overlap)
    pd.DataFrame([{"diagnostic": "near_neighbor_opposite_label_rate", "value": near_opposite_rate}, {"diagnostic": "exact_duplicate_groups", "value": int(len(duplicate_indices))}, {"diagnostic": "conflicting_duplicate_groups", "value": int(duplicate_conflicts)}]).to_csv(artifact_dir / "round3_class_overlap_analysis.csv", index=False)
    return {"segmented": segmented_row, "duplicate_groups": len(duplicate_indices), "conflicting_duplicates": duplicate_conflicts, "near_opposite_rate": near_opposite_rate, "score_overlap": overlap}
