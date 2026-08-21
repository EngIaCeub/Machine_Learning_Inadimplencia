"""Round 4 CatBoost and temporal representation experiments."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from credit_default.config import get_project_config
from credit_default.data.split import DatasetSplits, make_splits
from credit_default.features.credit_default_features import RAW_CATEGORICAL_COLUMNS, build_behavioral_features
from credit_default.features.preprocessing import fit_preprocessor, transform_features
from credit_default.features.temporal import TEMPORAL_FEATURE_GROUPS, build_temporal_trajectory_features
from credit_default.modeling.evaluate import build_threshold_search_table, evaluate_binary_classifier, get_positive_class_scores, select_best_threshold_row
from credit_default.modeling.round2 import ARTIFACT_DIR, _model, _write_json


ROUND4_ARTIFACT_DIR = ARTIFACT_DIR
A3_GROUPS = ("round1", "bill", "payment")
CATBOOST_CONFIGS = (
    {"iterations": 300, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 5.0, "random_strength": 1.0, "bagging_temperature": 1.0},
    {"iterations": 300, "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 8.0, "random_strength": 0.5, "bagging_temperature": 1.0},
    {"iterations": 400, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 8.0, "random_strength": 0.5, "bagging_temperature": 1.0},
    {"iterations": 250, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0, "random_strength": 1.0, "bagging_temperature": 1.0},
)


def _xgb_builder(temporal_groups: tuple[str, ...] = ()):
    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        result = build_behavioral_features(frame, enabled_groups=A3_GROUPS)
        if temporal_groups:
            temporal = build_temporal_trajectory_features(frame, enabled_groups=temporal_groups)
            temporal_columns = [column for column in temporal.columns if column not in frame.columns]
            result = pd.concat([result, temporal.loc[:, temporal_columns]], axis=1)
        return result
    return builder


def _precomputed_builder(engineered: pd.DataFrame):
    """Reuse target-free feature construction while preserving fold preprocessing."""

    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        return engineered.loc[frame.index].copy()

    return builder


def _catboost_frame(frame: pd.DataFrame, temporal_groups: tuple[str, ...] = (), engineered: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    result = (engineered.loc[frame.index] if engineered is not None else _xgb_builder(temporal_groups)(frame)).copy()
    categorical = [column for column in RAW_CATEGORICAL_COLUMNS if column in result.columns]
    for column in categorical:
        result[column] = result[column].astype(str)
    return result, categorical


def _fast_threshold(y: pd.Series, scores: np.ndarray, objective: str = "f1") -> float:
    """Select the same rounded PR thresholds without repeated sklearn calls."""
    truth = np.asarray(y, dtype=np.int8)
    values = np.asarray(scores, dtype=float)
    _, _, pr_thresholds = precision_recall_curve(truth, values)
    thresholds = np.unique(np.concatenate(([0.5], np.round(pr_thresholds, 6))))
    order = np.argsort(-values, kind="mergesort")
    sorted_scores = values[order]
    cumulative_positive = np.cumsum(truth[order], dtype=np.int64)
    positives = int(truth.sum())
    rows: list[tuple[float, float, float, float]] = []
    for threshold in thresholds:
        count = int(np.searchsorted(-sorted_scores, -threshold, side="right"))
        tp = int(cumulative_positive[count - 1]) if count else 0
        fp = count - tp
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((float(threshold), f1, precision, recall))
    valid = [row for row in rows if row[3] >= 0.60]
    candidates = valid or rows
    if objective == "f1":
        return max(candidates, key=lambda row: (row[1], row[2], row[0]))[0]
    return max(candidates, key=lambda row: (row[2], row[1], row[0]))[0]


def _metrics(y: pd.Series, scores: np.ndarray, objective: str = "f1") -> tuple[dict[str, Any], pd.DataFrame]:
    threshold = _fast_threshold(y, scores, objective=objective)
    metrics = evaluate_binary_classifier(y, scores, threshold)
    return metrics, pd.DataFrame([metrics])


def _record(experiment: str, phase: str, model: str, feature_set: str, metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"experiment_id": experiment, "phase": phase, "model": model, "feature_set": feature_set, "auc": float(metrics["roc_auc"]), "precision": float(metrics["precision_positive"]), "recall": float(metrics["recall_positive"]), "f1_binary": float(metrics["f1_binary"]), "f1_macro": float(metrics["f1_macro"]), "f1_weighted": float(metrics["f1_weighted"]), "threshold": float(metrics["threshold"]), "tp": int(metrics["tp"]), "fp": int(metrics["fp"]), "tn": int(metrics["tn"]), "fn": int(metrics["fn"]), "seed": 42, **extra}


def _write_recovery_metadata(splits: DatasetSplits, artifact_dir: Path) -> None:
    """Persist runtime and train/validation resume metadata without touching TEST."""
    try:
        import catboost
        catboost_version = catboost.__version__
    except Exception:  # pragma: no cover - environment diagnostic only.
        catboost_version = "unavailable"
    try:
        import xgboost
        xgboost_version = xgboost.__version__
    except Exception:  # pragma: no cover - environment diagnostic only.
        xgboost_version = "unavailable"
    _write_json(artifact_dir / "round4_runtime_profile.json", {
        "environment": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "gpu": "not used; CPU first",
        "catboost": catboost_version,
        "xgboost": xgboost_version,
        "cv_threads": 1,
        "model_threads": {"xgboost": 1, "catboost": 2},
        "subset_used_for_selection": False,
        "test_access": "none",
    })
    _write_json(artifact_dir / "round4_split_manifest.json", {
        "seed": 42,
        "train_rows": len(splits.X_train),
        "validation_rows": len(splits.X_validation),
        "train_indices": [int(index) for index in splits.X_train.index],
        "validation_indices": [int(index) for index in splits.X_validation.index],
        "test_access": "none",
    })


def _cat_model(params: dict[str, Any]) -> CatBoostClassifier:
    return CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=False, thread_count=2, allow_writing_files=False, **params)


def _catboost_oof(X: pd.DataFrame, y: pd.Series, params: dict[str, Any], temporal_groups: tuple[str, ...] = (), n_splits: int = 3, artifact_dir: Path | None = None, experiment_id: str = "catboost") -> np.ndarray:
    scores = np.full(len(X), np.nan)
    engineered = _xgb_builder(temporal_groups)(X)
    checkpoint = artifact_dir / f"round4_{experiment_id}_catboost_folds.csv" if artifact_dir else None
    completed = pd.read_csv(checkpoint) if checkpoint and checkpoint.exists() else pd.DataFrame()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_pos, holdout_pos) in enumerate(splitter.split(X, y)):
        if not completed.empty and int(fold) in set(completed["fold"].astype(int)):
            cached = completed.loc[completed["fold"].astype(int) == fold]
            scores[cached["row_pos"].astype(int).to_numpy()] = cached["score"].to_numpy()
            continue
        started = time.perf_counter()
        train_frame, cat_cols = _catboost_frame(X.iloc[train_pos], engineered=engineered)
        holdout_frame, _ = _catboost_frame(X.iloc[holdout_pos], engineered=engineered)
        model = _cat_model(params)
        model.fit(train_frame, y.iloc[train_pos], cat_features=cat_cols)
        scores[holdout_pos] = model.predict_proba(holdout_frame)[:, 1]
        if checkpoint:
            fold_metrics, _ = _metrics(y.iloc[holdout_pos], scores[holdout_pos])
            fold_rows = pd.DataFrame({"fold": fold, "row_pos": holdout_pos, "score": scores[holdout_pos]})
            fold_rows["auc"] = fold_metrics["roc_auc"]
            fold_rows["f1_binary"] = fold_metrics["f1_binary"]
            fold_rows["elapsed_seconds"] = time.perf_counter() - started
            fold_rows.to_csv(checkpoint, mode="a", header=not checkpoint.exists(), index=False)
    if np.isnan(scores).any():
        raise AssertionError("CatBoost OOF must cover every TRAIN row.")
    return scores


def _xgb_oof(X: pd.DataFrame, y: pd.Series, temporal_groups: tuple[str, ...], n_splits: int = 3, artifact_dir: Path | None = None, experiment_id: str = "temporal") -> np.ndarray:
    scores = np.full(len(X), np.nan)
    engineered = _xgb_builder(temporal_groups)(X)
    builder = _precomputed_builder(engineered)
    checkpoint = artifact_dir / f"round4_{experiment_id}_xgb_njobs1_folds.csv" if artifact_dir else None
    completed = pd.read_csv(checkpoint) if checkpoint and checkpoint.exists() else pd.DataFrame()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_pos, holdout_pos) in enumerate(splitter.split(X, y)):
        if not completed.empty and int(fold) in set(completed["fold"].astype(int)):
            cached = completed.loc[completed["fold"].astype(int) == fold]
            scores[cached["row_pos"].astype(int).to_numpy()] = cached["score"].to_numpy()
            continue
        started = time.perf_counter()
        bundle = fit_preprocessor(X.iloc[train_pos], feature_builder=builder)
        X_fold = transform_features(bundle, X.iloc[train_pos], feature_builder=builder)
        X_holdout = transform_features(bundle, X.iloc[holdout_pos], feature_builder=builder)
        model = _model().set_params(n_jobs=1)
        model.fit(X_fold, y.iloc[train_pos])
        scores[holdout_pos] = get_positive_class_scores(model, X_holdout)
        if checkpoint:
            fold_metrics, _ = _metrics(y.iloc[holdout_pos], scores[holdout_pos])
            fold_rows = pd.DataFrame({"fold": fold, "row_pos": holdout_pos, "score": scores[holdout_pos]})
            fold_rows["auc"] = fold_metrics["roc_auc"]
            fold_rows["f1_binary"] = fold_metrics["f1_binary"]
            fold_rows["elapsed_seconds"] = time.perf_counter() - started
            fold_rows.to_csv(checkpoint, mode="a", header=not checkpoint.exists(), index=False)
    if np.isnan(scores).any():
        raise AssertionError("Temporal OOF must cover every TRAIN row.")
    return scores


def _validation_xgb(X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, temporal_groups: tuple[str, ...]) -> tuple[Any, pd.DataFrame, pd.DataFrame, np.ndarray]:
    builder = _xgb_builder(temporal_groups)
    bundle = fit_preprocessor(X_train, feature_builder=builder)
    train_transformed = transform_features(bundle, X_train, feature_builder=builder)
    validation_transformed = transform_features(bundle, X_validation, feature_builder=builder)
    model = _model().set_params(n_jobs=1).fit(train_transformed, y_train)
    return model, bundle, validation_transformed, get_positive_class_scores(model, validation_transformed)


def run_round4(splits: DatasetSplits, artifact_dir: Path = ROUND4_ARTIFACT_DIR) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    baseline = {"model": "XGBoostClassifier", "features": "ROUND1+BILL+PAYMENT", "auc": 0.7816975032079083, "ap": 0.5567847707583915, "precision": 0.5008250825082509, "recall": 0.6100502512562814, "f1": 0.5500679655641142, "threshold": 0.461883, "fp": 605, "fn": 388, "test_access": "none"}
    _write_json(artifact_dir / "round4_baseline.json", baseline)

    cat_records: list[dict[str, Any]] = []
    for index, params in enumerate(CATBOOST_CONFIGS):
        oof_scores = _catboost_oof(splits.X_train, splits.y_train, params)
        oof_metrics, _ = _metrics(splits.y_train, oof_scores, objective="f1")
        cat_records.append(_record(f"C1_{index}", "4A", "CatBoostClassifier", "A3", oof_metrics, oof_auc=oof_metrics["roc_auc"], oof_precision=oof_metrics["precision_positive"], oof_recall=oof_metrics["recall_positive"], oof_f1=oof_metrics["f1_binary"], oof_threshold=oof_metrics["threshold"], hyperparameters=json.dumps(params), categorical_columns=",".join(RAW_CATEGORICAL_COLUMNS), validation_auc=None))
        pd.DataFrame(cat_records).to_csv(artifact_dir / "round4_catboost_oof_experiments.csv", index=False)
    cat_frame = pd.DataFrame(cat_records).sort_values(["oof_f1", "oof_precision", "oof_auc"], ascending=[False, False, False]).reset_index(drop=True)
    best_cat_oof = cat_frame.iloc[0].to_dict()
    best_cat_params = json.loads(best_cat_oof["hyperparameters"])
    cat_train, cat_columns = _catboost_frame(splits.X_train)
    cat_val, _ = _catboost_frame(splits.X_validation)
    cat_model = _cat_model(best_cat_params).fit(cat_train, splits.y_train, cat_features=cat_columns)
    cat_scores = cat_model.predict_proba(cat_val)[:, 1]
    cat_metrics = evaluate_binary_classifier(splits.y_validation, cat_scores, float(best_cat_oof["oof_threshold"]))
    cat_validation = _record("C1_frozen_validation", "4A", "CatBoostClassifier", "A3", cat_metrics, oof_f1=best_cat_oof["oof_f1"], hyperparameters=json.dumps(best_cat_params))
    _write_json(artifact_dir / "round4_catboost_best_configuration.json", {"params": best_cat_params, "oof": best_cat_oof, "validation": cat_validation, "dependency_version": "1.2.10", "test_access": "none"})

    temporal_records: list[dict[str, Any]] = []
    temporal_sets = {"T0_A3": (), "T1_PAY": ("pay_trajectory",), "T2_BILL_PAYMENT": ("bill_payment_trajectory",), "T3_SHORTFALL": ("shortfall_coverage", "utilization_interactions"), "T4_ALL": tuple(TEMPORAL_FEATURE_GROUPS)}
    for name, groups in temporal_sets.items():
        if not groups:
            oof_scores = _xgb_oof(splits.X_train, splits.y_train, ())
        else:
            oof_scores = _xgb_oof(splits.X_train, splits.y_train, groups)
        oof_metrics, _ = _metrics(splits.y_train, oof_scores, objective="f1")
        temporal_records.append(_record(name, "4B", "XGBoostClassifier", "A3+" + ("+".join(groups) if groups else "none"), oof_metrics, oof_auc=oof_metrics["roc_auc"], oof_precision=oof_metrics["precision_positive"], oof_recall=oof_metrics["recall_positive"], oof_f1=oof_metrics["f1_binary"], oof_threshold=oof_metrics["threshold"], new_feature_count=sum(len(TEMPORAL_FEATURE_GROUPS[group]) for group in groups)))
        pd.DataFrame(temporal_records).to_csv(artifact_dir / "round4_temporal_feature_experiments.csv", index=False)
    temporal_frame = pd.DataFrame(temporal_records).sort_values(["oof_f1", "oof_precision", "oof_auc"], ascending=[False, False, False]).reset_index(drop=True)
    _write_json(artifact_dir / "round4_temporal_feature_list.json", {key: list(value) for key, value in TEMPORAL_FEATURE_GROUPS.items()})
    best_temporal = temporal_frame.iloc[0].to_dict()
    best_temporal_name = str(best_temporal["experiment_id"])
    best_temporal_groups = temporal_sets[best_temporal_name]
    _, _, val_temporal_X, temporal_scores = _validation_xgb(splits.X_train, splits.y_train, splits.X_validation, best_temporal_groups)
    temporal_metrics = evaluate_binary_classifier(splits.y_validation, temporal_scores, float(best_temporal["oof_threshold"]))
    temporal_validation = _record("T_frozen_validation", "4B", "XGBoostClassifier", "A3+" + "+".join(best_temporal_groups), temporal_metrics, oof_f1=best_temporal["oof_f1"])

    cat_temporal = None
    if float(best_cat_oof["oof_f1"]) > float(cat_frame.loc[cat_frame["experiment_id"] == "C1_0", "oof_f1"].iloc[0]) or float(best_temporal["oof_f1"]) > float(temporal_frame.loc[temporal_frame["experiment_id"] == "T0_A3", "oof_f1"].iloc[0]) + 0.005:
        cat_train_temporal, cat_columns_temporal = _catboost_frame(splits.X_train, best_temporal_groups)
        cat_val_temporal, _ = _catboost_frame(splits.X_validation, best_temporal_groups)
        model_temporal = _cat_model(best_cat_params).fit(cat_train_temporal, splits.y_train, cat_features=cat_columns_temporal)
        cat_temporal_scores = model_temporal.predict_proba(cat_val_temporal)[:, 1]
        cat_temporal_metrics = evaluate_binary_classifier(splits.y_validation, cat_temporal_scores, float(best_cat_oof["oof_threshold"]))
        cat_temporal = _record("C1_temporal_validation", "4C", "CatBoostClassifier", "A3+" + "+".join(best_temporal_groups), cat_temporal_metrics, hyperparameters=json.dumps(best_cat_params))

    candidates = [baseline, cat_validation, temporal_validation]
    if cat_temporal is not None:
        candidates.append(cat_temporal)
    comparison = pd.DataFrame(candidates).sort_values(["f1", "precision", "auc"], ascending=[False, False, False]).reset_index(drop=True)
    comparison.to_csv(artifact_dir / "round4_model_comparison.csv", index=False)
    best = comparison.iloc[0].to_dict()
    _write_json(artifact_dir / "round4_best_configuration.json", {"best": best, "catboost_temporal": cat_temporal, "sequence_model": {"executed": False, "reason": "no PyTorch/TensorFlow dependency in project"}, "test_access": "none"})
    _write_json(artifact_dir / "round4_metrics_summary.json", {"baseline": baseline, "catboost": cat_validation, "temporal": temporal_validation, "catboost_temporal": cat_temporal, "comparison": comparison.to_dict(orient="records"), "test_access": "none"})
    return {"baseline": baseline, "catboost_oof": cat_frame, "catboost": cat_validation, "temporal_oof": temporal_frame, "temporal": temporal_validation, "catboost_temporal": cat_temporal, "comparison": comparison, "best": best, "artifact_dir": artifact_dir}


def run_round4_from_dataset(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    return run_round4(make_splits(features, target))


def run_round4_temporal_only(splits: DatasetSplits, artifact_dir: Path = ROUND4_ARTIFACT_DIR) -> dict[str, Any]:
    """Run the executable temporal branch after CatBoost runtime validation."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_recovery_metadata(splits, artifact_dir)
    temporal_sets = {"T0_A3": (), "T1_PAY": ("pay_trajectory",), "T2_BILL_PAYMENT": ("bill_payment_trajectory",), "T3_SHORTFALL": ("shortfall_coverage", "utilization_interactions"), "T4_ALL": tuple(TEMPORAL_FEATURE_GROUPS)}
    records: list[dict[str, Any]] = []
    for name, groups in temporal_sets.items():
        scores = _xgb_oof(splits.X_train, splits.y_train, groups, n_splits=3, artifact_dir=artifact_dir, experiment_id=name)
        metrics, _ = _metrics(splits.y_train, scores, objective="f1")
        records.append(_record(name, "4B", "XGBoostClassifier", "A3+" + ("+".join(groups) if groups else "none"), metrics, oof_auc=metrics["roc_auc"], oof_precision=metrics["precision_positive"], oof_recall=metrics["recall_positive"], oof_f1=metrics["f1_binary"], oof_threshold=metrics["threshold"], new_feature_count=sum(len(TEMPORAL_FEATURE_GROUPS[group]) for group in groups)))
        pd.DataFrame(records).to_csv(artifact_dir / "round4_temporal_feature_experiments.csv", index=False)
    frame = pd.DataFrame(records).sort_values(["oof_f1", "oof_precision", "oof_auc"], ascending=[False, False, False]).reset_index(drop=True)
    _write_json(artifact_dir / "round4_temporal_feature_list.json", {key: list(value) for key, value in TEMPORAL_FEATURE_GROUPS.items()})
    best_oof = frame.iloc[0].to_dict()
    groups = temporal_sets[str(best_oof["experiment_id"])]
    confirmation_scores = _xgb_oof(
        splits.X_train,
        splits.y_train,
        groups,
        n_splits=5,
        artifact_dir=artifact_dir,
        experiment_id=f"confirmation_{best_oof['experiment_id']}",
    )
    confirmation_metrics, _ = _metrics(splits.y_train, confirmation_scores)
    _write_json(artifact_dir / "round4_temporal_confirmation.json", _record("temporal_confirmation", "4B-confirmation", "XGBoostClassifier", "A3+" + "+".join(groups), confirmation_metrics, oof_f1=confirmation_metrics["f1_binary"]))
    _, _, validation_X, scores = _validation_xgb(splits.X_train, splits.y_train, splits.X_validation, groups)
    metrics = evaluate_binary_classifier(splits.y_validation, scores, float(confirmation_metrics["threshold"]))
    validation = _record("T_frozen_validation", "4B", "XGBoostClassifier", "A3+" + "+".join(groups), metrics, oof_f1=best_oof["oof_f1"])
    comparison = pd.DataFrame([{"model": "XGBoostClassifier", "features": "A3", **{key: value for key, value in {"auc": 0.7816975032079083, "precision": 0.5008250825082509, "recall": 0.6100502512562814, "f1": 0.5500679655641142}.items()}}, {"model": validation["model"], "features": validation["feature_set"], "auc": validation["auc"], "precision": validation["precision"], "recall": validation["recall"], "f1": validation["f1_binary"], "threshold": validation["threshold"], "fp": validation["fp"], "fn": validation["fn"]}])
    comparison.to_csv(artifact_dir / "round4_model_comparison.csv", index=False)
    _write_json(artifact_dir / "round4_metrics_summary.json", {"catboost": {"executed": False, "reason": "scheduled after temporal branch"}, "temporal_oof": frame.to_dict(orient="records"), "temporal_confirmation": confirmation_metrics, "temporal_validation": validation, "comparison": comparison.to_dict(orient="records"), "sequence_model": {"executed": False, "reason": "not in recovery scope"}, "test_access": "none"})
    _write_json(artifact_dir / "round4_best_configuration.json", {"best": validation, "catboost_executed": False, "test_access": "none"})
    return {"temporal_oof": frame, "temporal": validation, "comparison": comparison, "best": validation, "artifact_dir": artifact_dir}


def run_round4_temporal_only_from_dataset(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    return run_round4_temporal_only(make_splits(features, target))
