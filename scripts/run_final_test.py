"""One-time final holdout evaluation for the frozen A1 development winner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from credit_default.data.load import load_uci_dataset
from credit_default.data.split import make_splits
from credit_default.modeling.evaluate import evaluate_binary_classifier
from credit_default.modeling.round4 import _catboost_frame
from credit_default.modeling.round2 import _write_json


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "final"
CONFIG_PATH = FINAL / "development_winner.json"
MARKER_PATH = FINAL / "final_test_access_marker.json"


def _feature_hash(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _validate_freeze(config: dict[str, object]) -> bool:
    metrics = config["metrics"]
    if config.get("model") != "CatBoostClassifier":
        raise RuntimeError("FINAL_TEST_BLOCKED_FREEZE_MODEL")
    if config.get("feature_set") != "A3":
        raise RuntimeError("FINAL_TEST_BLOCKED_FREEZE_FEATURES")
    if config.get("official_f1_metric") != "macro_f1":
        raise RuntimeError("FINAL_TEST_BLOCKED_FREEZE_METRIC")
    if config.get("positive_class", {}).get("value") != 1:
        raise RuntimeError("FINAL_TEST_BLOCKED_FREEZE_POSITIVE_CLASS")
    if abs(float(metrics["threshold"]) - 0.247743) > 1e-12:
        raise RuntimeError("FINAL_TEST_BLOCKED_FREEZE_THRESHOLD")
    if config.get("methodology_correction_applied") is not True:
        raise RuntimeError("FINAL_TEST_BLOCKED_FREEZE_CORRECTION")
    if config.get("all_a1_gates_pass") is not True:
        raise RuntimeError("FINAL_TEST_BLOCKED_VALIDATION_GATES")
    recovery = MARKER_PATH.exists()
    if recovery and os.environ.get("ALLOW_FINAL_TEST_RECOVERY") != "1":
        raise RuntimeError("FINAL_TEST_ALREADY_EVALUATED")
    return recovery


def _check_split_manifest(splits: object) -> dict[str, int]:
    manifest = json.loads((ROOT / "artifacts" / "experiments" / "round4_split_manifest.json").read_text(encoding="utf-8"))
    train = set(int(index) for index in manifest["train_indices"])
    validation = set(int(index) for index in manifest["validation_indices"])
    actual_train = set(int(index) for index in splits.X_train.index)
    actual_validation = set(int(index) for index in splits.X_validation.index)
    actual_test = set(int(index) for index in splits.X_test.index)
    if train != actual_train or validation != actual_validation:
        raise RuntimeError("FINAL_TEST_BLOCKED_SPLIT_MANIFEST_MISMATCH")
    if train & validation or train & actual_test or validation & actual_test:
        raise RuntimeError("FINAL_TEST_BLOCKED_SPLIT_OVERLAP")
    return {"train_rows": len(train), "validation_rows": len(validation), "test_rows": len(actual_test)}


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    recovery = _validate_freeze(config)
    access_count = 2 if recovery else 1
    features, target = load_uci_dataset()
    splits = make_splits(features, target)
    split_sizes = _check_split_manifest(splits)

    train_frame, categorical_columns = _catboost_frame(splits.X_train)
    test_frame, test_categorical_columns = _catboost_frame(splits.X_test)
    if list(train_frame.columns) != list(test_frame.columns):
        raise RuntimeError("FINAL_TEST_BLOCKED_FEATURE_ORDER")
    if categorical_columns != test_categorical_columns:
        raise RuntimeError("FINAL_TEST_BLOCKED_CATEGORICAL_MAPPING")
    feature_columns = list(train_frame.columns)
    feature_audit = {"number_features": len(feature_columns), "feature_order_hash": _feature_hash(feature_columns), "missing_columns": [], "extra_columns": [], "train_dtypes": train_frame.dtypes.astype(str).to_dict(), "test_dtypes": test_frame.dtypes.astype(str).to_dict()}

    params = dict(config["hyperparameters"])
    model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", random_seed=int(config["seed"]), verbose=False, thread_count=2, allow_writing_files=False, **params)
    model.fit(train_frame, splits.y_train, cat_features=categorical_columns)

    # This marker is written immediately before the only TEST prediction call.
    _write_json(MARKER_PATH, {"final_test_access_count": access_count, "recovery_count": int(recovery), "recovery_reason": "deterministic evaluator missing micro_f1" if recovery else None, "model": "CatBoostClassifier", "feature_set": "A3", "threshold": float(config["metrics"]["threshold"]), "test_access_started": True})
    test_scores = model.predict_proba(test_frame)[:, 1]
    threshold = float(config["metrics"]["threshold"])
    test_metrics = evaluate_binary_classifier(splits.y_test, test_scores, threshold)
    test_metrics["average_precision"] = float(average_precision_score(splits.y_test, test_scores))
    test_metrics["feature_order_hash"] = feature_audit["feature_order_hash"]
    test_metrics["split_sizes"] = split_sizes
    gates = {"auc_075": float(test_metrics["roc_auc"]) >= 0.75, "recall_060": float(test_metrics["recall_positive"]) >= 0.60, "macro_f1_065": float(test_metrics["f1_macro"]) >= 0.65}

    predictions = pd.DataFrame({"row_identifier": splits.X_test.index.to_numpy(), "y_true": splits.y_test.to_numpy(), "probability_default_1": test_scores, "frozen_threshold": threshold, "y_pred": (test_scores >= threshold).astype(int)})
    predictions.to_csv(FINAL / "final_test_predictions.csv", index=False)
    _write_json(FINAL / "final_test_metrics.json", {"methodology_version": "A1_MACRO_F1", "model": "CatBoostClassifier", "feature_set": "A3", "positive_class": 1, "threshold": threshold, "metrics": {key: test_metrics[key] for key in ("roc_auc", "average_precision", "precision_positive", "recall_positive", "f1_binary_default_1", "f1_class_0", "f1_macro", "f1_weighted", "accuracy", "micro_f1")}, "confusion_matrix": {key: int(test_metrics[key]) for key in ("tn", "fp", "fn", "tp")}, "gates": gates, "overall_pass": all(gates.values()), "feature_audit": feature_audit, "test_access_count": access_count, "deterministic_recovery": recovery})
    _write_json(FINAL / "final_test_gate_status.json", {"status": "A1_FINAL_TEST_GATES_REACHED" if all(gates.values()) else "A1_FINAL_TEST_GATES_NOT_REACHED", "gates": gates, "overall_pass": all(gates.values()), "test_access_count": access_count, "deterministic_recovery": recovery})

    validation = config["metrics"]
    comparison = pd.DataFrame([{"metric": "AUC", "validation": validation["roc_auc"], "test": test_metrics["roc_auc"], "delta_test_minus_validation": test_metrics["roc_auc"] - validation["roc_auc"]}, {"metric": "Precision", "validation": validation["precision"], "test": test_metrics["precision_positive"], "delta_test_minus_validation": test_metrics["precision_positive"] - validation["precision"]}, {"metric": "Recall", "validation": validation["recall"], "test": test_metrics["recall_positive"], "delta_test_minus_validation": test_metrics["recall_positive"] - validation["recall"]}, {"metric": "Binary F1", "validation": validation["f1_binary"], "test": test_metrics["f1_binary_default_1"], "delta_test_minus_validation": test_metrics["f1_binary_default_1"] - validation["f1_binary"]}, {"metric": "Macro F1", "validation": validation["f1_macro"], "test": test_metrics["f1_macro"], "delta_test_minus_validation": test_metrics["f1_macro"] - validation["f1_macro"]}, {"metric": "Weighted F1", "validation": validation["f1_weighted"], "test": test_metrics["f1_weighted"], "delta_test_minus_validation": test_metrics["f1_weighted"] - validation["f1_weighted"]}])
    comparison.to_csv(FINAL / "final_validation_vs_test.csv", index=False)
    _write_json(FINAL / "final_model_evaluation.json", {"development_validation": validation, "final_test": test_metrics, "generalization_delta": comparison.to_dict(orient="records"), "post_test_tuning": False})

    tn, fp, fn, tp = (int(test_metrics[key]) for key in ("tn", "fp", "fn", "tp"))
    plt.figure(figsize=(6, 5)); plt.imshow([[tn, fp], [fn, tp]], cmap="Blues"); plt.colorbar(); plt.xticks([0, 1], ["Pred 0", "Pred 1"]); plt.yticks([0, 1], ["True 0", "True 1"]); plt.title(f"FINAL TEST | Frozen CatBoost A3 | Threshold={threshold}"); plt.tight_layout(); plt.savefig(FINAL / "test_confusion_matrix.png", dpi=140); plt.close()
    fpr, tpr, _ = roc_curve(splits.y_test, test_scores); plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr, label=f"TEST AUC={test_metrics['roc_auc']:.4f}"); plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout(); plt.savefig(FINAL / "test_roc_curve.png", dpi=140); plt.close()
    precision, recall, _ = precision_recall_curve(splits.y_test, test_scores); plt.figure(figsize=(6, 5)); plt.plot(recall, precision, label=f"TEST AP={test_metrics['average_precision']:.4f}"); plt.scatter([test_metrics["recall_positive"]], [test_metrics["precision_positive"]], label="Frozen threshold"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout(); plt.savefig(FINAL / "test_precision_recall_curve.png", dpi=140); plt.close()

    audit = json.loads((FINAL / "leakage_audit.json").read_text(encoding="utf-8"))
    audit.update({"final_test_accessed": True, "final_test_access_count": access_count, "deterministic_infrastructure_recovery": recovery, "recovery_reason": "missing micro_f1 output key" if recovery else None, "model_frozen_before_test": True, "threshold_frozen_before_test": True, "feature_set_frozen_before_test": True, "metric_definition_frozen_before_test": True, "test_used_for_model_selection": False, "test_used_for_threshold_tuning": False, "test_used_for_feature_selection": False})
    _write_json(FINAL / "leakage_audit.json", audit)
    summary = FINAL / "final_evaluation_summary.md"
    summary.write_text(f"""# Final Evaluation\n\nFrozen candidate: CatBoost A3. Official F1: Macro F1. Positive class: `default=1`. Threshold: `{threshold}`.\n\nValidation: AUC `{validation['roc_auc']:.4f}`, Recall `{validation['recall']:.4f}`, Macro F1 `{validation['f1_macro']:.4f}`.\n\nTEST: AUC `{test_metrics['roc_auc']:.4f}`, AP `{test_metrics['average_precision']:.4f}`, Precision `{test_metrics['precision_positive']:.4f}`, Recall `{test_metrics['recall_positive']:.4f}`, Binary F1 `{test_metrics['f1_binary_default_1']:.4f}`, Macro F1 `{test_metrics['f1_macro']:.4f}`, Weighted F1 `{test_metrics['f1_weighted']:.4f}`.\n\nGates: AUC `{gates['auc_075']}`, Recall `{gates['recall_060']}`, Macro F1 `{gates['macro_f1_065']}`. Overall: `{all(gates.values())}`.\n\nThe final TEST set was evaluated only after model, feature set, threshold and metric definition had been frozen. No model selection, threshold optimization or hyperparameter tuning was performed using TEST results.\n""", encoding="utf-8")
    print(json.dumps({"status": "A1_FINAL_TEST_GATES_REACHED" if all(gates.values()) else "A1_FINAL_TEST_GATES_NOT_REACHED", "metrics": test_metrics, "gates": gates, "split_sizes": split_sizes}, indent=2, default=str))


if __name__ == "__main__":
    main()
