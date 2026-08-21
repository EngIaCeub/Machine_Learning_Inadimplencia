"""Profile the Round 4 T1 fold without selecting a model or touching TEST."""

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from credit_default.data.load import load_uci_dataset
from credit_default.data.split import make_splits
from credit_default.features.credit_default_features import build_behavioral_features
from credit_default.features.temporal import build_temporal_trajectory_features
from credit_default.features.preprocessing import fit_preprocessor, transform_features
from credit_default.modeling.evaluate import build_threshold_search_table, evaluate_binary_classifier, get_positive_class_scores, select_best_threshold_row
from credit_default.modeling.round2 import ARTIFACT_DIR, _model, _write_json
from credit_default.modeling.round4 import A3_GROUPS, _fast_threshold


def _engineer(frame: pd.DataFrame, temporal: bool) -> pd.DataFrame:
    base = build_behavioral_features(frame, enabled_groups=A3_GROUPS)
    if not temporal:
        return base
    extra = build_temporal_trajectory_features(frame, enabled_groups=("pay_trajectory",))
    extra = extra.loc[:, [c for c in extra.columns if c not in frame.columns]]
    return pd.concat([base, extra], axis=1).copy()


def _audit(name: str, frame: pd.DataFrame) -> dict[str, object]:
    numeric = frame.select_dtypes(include=["number"])
    counts = frame.dtypes.astype(str).value_counts().to_dict()
    non_finite = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()) if not numeric.empty else 0
    nunique = frame.nunique(dropna=False)
    return {
        "feature_set": name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "memory_mb": float(frame.memory_usage(deep=True).sum() / 1024**2),
        "number_numeric": int(frame.select_dtypes(include=["number"]).shape[1]),
        "number_integer": int(frame.select_dtypes(include=["integer"]).shape[1]),
        "number_float": int(frame.select_dtypes(include=["floating"]).shape[1]),
        "number_object": int(frame.select_dtypes(include=["object"]).shape[1]),
        "number_category": int(frame.select_dtypes(include=["category"]).shape[1]),
        "number_bool": int(frame.select_dtypes(include=["bool"]).shape[1]),
        "dtypes_summary": counts,
        "nan": int(frame.isna().sum().sum()),
        "inf": non_finite,
        "duplicate_column_names": int(frame.columns.duplicated().sum()),
        "constant_columns": int((nunique <= 1).sum()),
        "near_constant_columns": int((nunique <= max(2, int(len(frame) * 0.001))).sum()),
        "duplicate_features": int(frame.T.duplicated().sum()),
        "fragmentation_blocks": int(getattr(frame, "_mgr", getattr(frame, "_data", None)).nblocks),
        "max_abs_value": float(np.nanmax(np.abs(numeric.to_numpy(dtype=float)))) if not numeric.empty else 0.0,
    }


def _profile_one(X_train: pd.DataFrame, y_train: pd.Series, train_pos: np.ndarray, holdout_pos: np.ndarray, X_t1: pd.DataFrame, label: str, n_jobs: int, input_timing: dict[str, float]) -> dict[str, object]:
    started = time.perf_counter()
    raw_train = X_train.iloc[train_pos]
    raw_holdout = X_train.iloc[holdout_pos]
    builder = lambda frame: X_t1.loc[frame.index].copy()
    t0 = time.perf_counter()
    bundle = fit_preprocessor(raw_train, feature_builder=builder)
    X_fold = transform_features(bundle, raw_train, feature_builder=builder)
    X_holdout = transform_features(bundle, raw_holdout, feature_builder=builder)
    preprocessing_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    X_fold_np = X_fold.to_numpy(dtype=np.float32, copy=True)
    X_holdout_np = X_holdout.to_numpy(dtype=np.float32, copy=True)
    input_seconds = time.perf_counter() - t1
    model = _model().set_params(n_jobs=n_jobs)
    params = model.get_params()
    t2 = time.perf_counter()
    model.fit(X_fold_np, y_train.iloc[train_pos])
    fit_seconds = time.perf_counter() - t2
    t3 = time.perf_counter()
    scores = get_positive_class_scores(model, X_holdout_np)
    predict_seconds = time.perf_counter() - t3
    t4 = time.perf_counter()
    threshold = _fast_threshold(y_train.iloc[holdout_pos], scores)
    threshold_seconds = time.perf_counter() - t4
    t5 = time.perf_counter()
    metrics = evaluate_binary_classifier(y_train.iloc[holdout_pos], scores, threshold)
    metrics_seconds = time.perf_counter() - t5
    return {
        "profile": label,
        "n_jobs": n_jobs,
        "runtime_seconds": time.perf_counter() - started,
        "preprocessing_seconds": preprocessing_seconds,
        "d_matrix_or_xgb_input_seconds": input_seconds,
        "model_fit_seconds": fit_seconds,
        "predict_proba_seconds": predict_seconds,
        "threshold_search_seconds": threshold_seconds,
        "metrics_seconds": metrics_seconds,
        "roc_auc": float(metrics["roc_auc"]),
        "f1_binary": float(metrics["f1_binary"]),
        "recall": float(metrics["recall_positive"]),
        "precision": float(metrics["precision_positive"]),
        "input_rows": int(len(X_fold_np)),
        "input_columns": int(X_fold_np.shape[1]),
        "input_memory_mb": float((X_fold_np.nbytes + X_holdout_np.nbytes) / 1024**2),
        "xgboost": {key: params.get(key) for key in ("tree_method", "device", "n_jobs", "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "scale_pos_weight")},
        "seed": 42,
    }


def main() -> None:
    artifact_dir = ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    features, target = load_uci_dataset()
    splits = make_splits(features, target)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    train_pos, holdout_pos = next(splitter.split(splits.X_train, splits.y_train))

    t0 = time.perf_counter()
    t0_frame = _engineer(splits.X_train, temporal=False)
    t0_build_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    t1_frame = _engineer(splits.X_train, temporal=True)
    t1_build_seconds = time.perf_counter() - t1
    audits = pd.DataFrame([_audit("T0_A3", t0_frame), _audit("T1_A3_PAY", t1_frame)])
    audits.to_csv(artifact_dir / "round4_feature_matrix_audit.csv", index=False)

    profiles = []
    for label, n_jobs in (("P1_n_jobs_1", 1), ("P2_n_jobs_moderate", min(4, 8)), ("P3_n_jobs_current", 2)):
        profiles.append(_profile_one(splits.X_train, splits.y_train, train_pos, holdout_pos, t1_frame, label, n_jobs, {}))
        pd.DataFrame(profiles).to_csv(artifact_dir / "round4_thread_profile.csv", index=False)

    _write_json(artifact_dir / "round4_runtime_diagnostic.json", {
        "status": "COMPLETED_FOLD_0_ONLY",
        "environment": platform.platform(),
        "python": platform.python_version(),
        "input_rows": len(splits.X_train),
        "fold_train_rows": len(train_pos),
        "fold_holdout_rows": len(holdout_pos),
        "feature_construction_seconds": {"T0": t0_build_seconds, "T1": t1_build_seconds},
        "feature_audit": audits.to_dict(orient="records"),
        "thread_profiles": profiles,
        "test_access": "none",
        "subset_used_for_selection": False,
    })
    print(json.dumps({"feature_audit": audits.to_dict(orient="records"), "thread_profiles": profiles}, indent=2))


if __name__ == "__main__":
    main()
