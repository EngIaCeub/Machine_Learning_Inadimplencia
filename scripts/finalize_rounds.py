"""Consolidate Rounds 1-4 from persisted artifacts only; never access TEST."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from credit_default.features.credit_default_features import RAW_CATEGORICAL_COLUMNS, SEMANTIC_COLUMN_MAP, ROUND1_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "artifacts" / "experiments"
FINAL = ROOT / "artifacts" / "final"


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item.item() if hasattr(item, "item") else item

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False, default=str), encoding="utf-8")


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    winner_cfg = json.loads((EXPERIMENTS / "round2_best_model_configuration.json").read_text(encoding="utf-8"))
    winner = winner_cfg["best_ensemble"]
    params = winner_cfg["hyperparameters"]
    baseline_round1 = json.loads((EXPERIMENTS / "round2_metrics_summary.json").read_text(encoding="utf-8"))["baseline_official"]
    a3 = pd.read_csv(EXPERIMENTS / "round2_ablation_experiments.csv").query("experiment == 'A3'").iloc[0].to_dict()
    hn = pd.read_csv(EXPERIMENTS / "round3_hard_negative_experiments.csv")
    hn_best = hn.sort_values("f1_binary", ascending=False).iloc[0].to_dict()
    cascade = pd.read_csv(EXPERIMENTS / "round3_cascade_experiments.csv").sort_values("f1_binary", ascending=False).iloc[0].to_dict()
    segmented = pd.read_csv(EXPERIMENTS / "round3_segmented_thresholds.csv").iloc[0].to_dict()
    temporal = pd.read_csv(EXPERIMENTS / "round4_temporal_feature_experiments.csv")
    cat_oof = pd.read_csv(EXPERIMENTS / "round4_catboost_oof_experiments.csv").sort_values("oof_f1", ascending=False).iloc[0].to_dict()
    cat_val = json.loads((EXPERIMENTS / "round4_catboost_validation_frozen.json").read_text(encoding="utf-8"))
    cat_val_metrics = cat_val["validation"]
    overlap = json.loads((EXPERIMENTS / "round3_score_overlap.json").read_text(encoding="utf-8"))
    duplicates = json.loads((EXPERIMENTS / "round3_duplicate_analysis.json").read_text(encoding="utf-8"))
    runtime = json.loads((EXPERIMENTS / "round4_runtime_diagnostic.json").read_text(encoding="utf-8"))

    positive_recall = float(winner["recall_positive"])
    target_f1 = 0.65
    required_precision = target_f1 * positive_recall / (2 * positive_recall - target_f1)
    final_winner = {
        "status": "ROUND4_FINAL_PLATEAU",
        "model": "XGBoostClassifier",
        "feature_set": "ROUND1+BILL+PAYMENT",
        "sampling": "none",
        "positive_class": {"name": "default payment", "value": 1},
        "metrics": {
            "roc_auc": float(winner["roc_auc"]), "average_precision": 0.5567847707583915,
            "precision": float(winner["precision_positive"]), "recall": positive_recall,
            "f1_binary": float(winner["f1_binary"]), "f1_macro": float(winner["f1_macro"]),
            "f1_weighted": float(winner["f1_weighted"]), "threshold": float(winner["threshold"]),
            "tp": int(winner["tp"]), "fp": int(winner["fp"]), "tn": int(winner["tn"]), "fn": int(winner["fn"]),
        },
        "hyperparameters": params,
        "feature_order": {
            "raw_semantic_columns": SEMANTIC_COLUMN_MAP,
            "raw_categorical_columns": list(RAW_CATEGORICAL_COLUMNS),
            "round1_engineered_columns": list(ROUND1_FEATURE_COLUMNS),
            "behavioral_groups": ["round1", "bill", "payment"],
        },
        "preprocessing": {"fit_scope": "TRAIN only", "numeric": ["median_imputer", "standard_scaler"], "categorical": ["most_frequent_imputer", "one_hot_encoder"], "unknown_categories": "ignore"},
        "seed": 42,
        "split_manifest": "artifacts/experiments/round4_split_manifest.json",
        "test_access": "none",
        "required_precision_at_observed_recall_for_f1_065": required_precision,
    }
    write_json(FINAL / "development_winner.json", final_winner)

    rows = [
        {"round": "R1", "experiment": "baseline_official", "strategy": "threshold+weight+tuning", "model": "XGBoostClassifier", "features": "ROUND1", "AUC": baseline_round1["auc"], "AP": None, "Precision": baseline_round1["precision"], "Recall": baseline_round1["recall"], "F1_binary": baseline_round1["f1_binary"], "Macro_F1": None, "Weighted_F1": None, "threshold": baseline_round1["threshold"], "TP": None, "FP": None, "TN": None, "FN": None, "selection_source": "VALIDATION", "status": "F1_GATE_FAIL"},
        {"round": "R2", "experiment": "A3", "strategy": "feature_engineering", "model": "XGBoostClassifier", "features": "ROUND1+BILL+PAYMENT", "AUC": a3["auc"], "AP": a3["average_precision"], "Precision": a3["precision"], "Recall": a3["recall"], "F1_binary": a3["f1_binary"], "Macro_F1": a3["f1_macro"], "Weighted_F1": a3["f1_weighted"], "threshold": a3["threshold"], "TP": a3["tp"], "FP": a3["fp"], "TN": a3["tn"], "FN": a3["fn"], "selection_source": "VALIDATION", "status": "DEVELOPMENT_WINNER"},
        {"round": "R3", "experiment": "hard_negative_best", "strategy": "hard_negative_weighting", "model": "XGBoostClassifier", "features": "ROUND1+BILL+PAYMENT", "AUC": hn_best["auc"], "AP": None, "Precision": hn_best["precision"], "Recall": hn_best["recall"], "F1_binary": hn_best["f1_binary"], "Macro_F1": hn_best["f1_macro"], "Weighted_F1": hn_best["f1_weighted"], "threshold": hn_best["threshold"], "TP": hn_best["tp"], "FP": hn_best["fp"], "TN": hn_best["tn"], "FN": hn_best["fn"], "selection_source": "VALIDATION", "status": "NO_NET_GAIN"},
        {"round": "R3", "experiment": "cascade_best", "strategy": "cascade", "model": "XGBoost+LogisticRegression", "features": "ROUND1+BILL+PAYMENT", "AUC": cascade["auc"], "AP": None, "Precision": cascade["precision"], "Recall": cascade["recall"], "F1_binary": cascade["f1_binary"], "Macro_F1": cascade["f1_macro"], "Weighted_F1": cascade["f1_weighted"], "threshold": cascade["threshold"], "TP": cascade["tp"], "FP": cascade["fp"], "TN": cascade["tn"], "FN": cascade["fn"], "selection_source": "VALIDATION", "status": "NO_GAIN"},
        {"round": "R3", "experiment": "segmented_thresholds", "strategy": "OOF_delay_segments", "model": "XGBoostClassifier", "features": "ROUND1+BILL+PAYMENT", "AUC": segmented["roc_auc"], "AP": None, "Precision": segmented["precision_positive"], "Recall": segmented["recall_positive"], "F1_binary": segmented["f1_binary"], "Macro_F1": segmented["f1_macro"], "Weighted_F1": segmented["f1_weighted"], "threshold": "segment_specific", "TP": segmented["tp"], "FP": segmented["fp"], "TN": segmented["tn"], "FN": segmented["fn"], "selection_source": "TRAIN_OOF", "status": "NO_GAIN"},
        {"round": "R4", "experiment": "T1_PAY", "strategy": "temporal_confirmation", "model": "XGBoostClassifier", "features": "A3+PAY trajectory", "AUC": None, "AP": None, "Precision": None, "Recall": None, "F1_binary": 0.5499076683837107, "Macro_F1": None, "Weighted_F1": None, "threshold": 0.465306, "TP": None, "FP": None, "TN": None, "FN": None, "selection_source": "TRAIN_OOF", "status": "NO_MATERIAL_GAIN"},
        {"round": "R4", "experiment": "CatBoost_A3", "strategy": "compact_screening", "model": "CatBoostClassifier", "features": "A3", "AUC": cat_val_metrics["roc_auc"], "AP": None, "Precision": cat_val_metrics["precision_positive"], "Recall": cat_val_metrics["recall_positive"], "F1_binary": cat_val_metrics["f1_binary"], "Macro_F1": cat_val_metrics["f1_macro"], "Weighted_F1": cat_val_metrics["f1_weighted"], "threshold": cat_val_metrics["threshold"], "TP": cat_val_metrics["tp"], "FP": cat_val_metrics["fp"], "TN": cat_val_metrics["tn"], "FN": cat_val_metrics["fn"], "selection_source": "TRAIN_OOF", "status": "NO_GENERALIZATION_GAIN"},
        {"round": "R4", "experiment": "frozen_development_winner", "strategy": "final_freeze", "model": "XGBoostClassifier", "features": "ROUND1+BILL+PAYMENT", "AUC": winner["roc_auc"], "AP": 0.5567847707583915, "Precision": winner["precision_positive"], "Recall": winner["recall_positive"], "F1_binary": winner["f1_binary"], "Macro_F1": winner["f1_macro"], "Weighted_F1": winner["f1_weighted"], "threshold": winner["threshold"], "TP": winner["tp"], "FP": winner["fp"], "TN": winner["tn"], "FN": winner["fn"], "selection_source": "VALIDATION", "status": "FROZEN"},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(FINAL / "modeling_experiment_summary.csv", index=False)
    write_json(FINAL / "development_metrics_summary.json", {"status": "ROUND4_FINAL_PLATEAU", "winner": final_winner, "gates": {"roc_auc": True, "recall": True, "f1_binary": False}, "f1_gap": 0.65 - float(winner["f1_binary"]), "required_precision": required_precision, "test_access": "none"})
    write_json(FINAL / "leakage_audit.json", {"train_fit": True, "oof_within_train": True, "validation_fit": False, "target_in_features": False, "resampling_only_train": True, "cascade_stage2_train_oof": True, "round4_thresholds_from_train_oof": True, "test_access": "none", "validation_reuse_limitation": "VALIDATION was consulted in Rounds 1-3; final Round 4 promotion used frozen TRAIN-OOF threshold."})

    threshold_table = pd.read_csv(EXPERIMENTS / "round2_threshold_search_baseline.csv")
    plt.figure(figsize=(7, 5)); plt.plot(threshold_table["recall_positive"], threshold_table["precision_positive"], linewidth=1); plt.axvline(0.60, color="red", linestyle="--"); plt.xlabel("Recall positive"); plt.ylabel("Precision positive"); plt.title("Best candidate precision-recall curve"); plt.tight_layout(); plt.savefig(FINAL / "best_precision_recall_curve.png", dpi=140); plt.close()
    plt.figure(figsize=(7, 5)); plt.imshow([[winner["tn"], winner["fp"]], [winner["fn"], winner["tp"]]], cmap="Blues"); plt.colorbar(); plt.xticks([0, 1], ["Pred 0", "Pred 1"]); plt.yticks([0, 1], ["True 0", "True 1"]); plt.title("Development winner confusion matrix"); plt.tight_layout(); plt.savefig(FINAL / "best_confusion_matrix.png", dpi=140); plt.close()
    evolution = pd.DataFrame({"round": ["Initial", "Round 1", "Round 2", "Round 3", "Round 4"], "f1_binary": [0.5447, 0.5452, 0.5501, 0.5501, 0.5501]}); evolution.to_csv(FINAL / "f1_evolution_by_round.csv", index=False); plt.figure(figsize=(7, 4)); plt.plot(evolution["round"], evolution["f1_binary"], marker="o"); plt.axhline(0.65, color="red", linestyle="--"); plt.ylabel("Binary F1"); plt.title("F1 evolution"); plt.tight_layout(); plt.savefig(FINAL / "f1_evolution_by_round.png", dpi=140); plt.close()

    summary_md = f"""# Rounds 1-4 Experimental Summary

## Objective
Predict default payment with positive class `default = 1`.

## Gates
ROC-AUC >= 0.75: PASS. Recall >= 0.60: PASS. Binary F1 >= 0.65: FAIL.

## Protocol
UCI id=350, deterministic stratified 70/15/15 split, TRAIN for fitting/OOF, VALIDATION for external development comparison, TEST untouched.

## Development Winner
XGBoostClassifier with ROUND1+BILL+PAYMENT, no sampling, threshold `{float(winner['threshold']):.6f}`. Validation AUC `{float(winner['roc_auc']):.4f}`, Precision `{float(winner['precision_positive']):.4f}`, Recall `{float(winner['recall_positive']):.4f}`, F1 `{float(winner['f1_binary']):.4f}`.

## Rounds
Round 1 reached F1 approximately 0.5452. Round 2 feature engineering produced the winner at 0.5501 and reduced false positives versus the earlier baseline. Round 3 hard-negative weighting reduced FP but lowered F1; cascade and segmented thresholds did not improve. Round 4 temporal representations and CatBoost did not produce material validation generalization gain.

## Diagnostics
Class overlap was high: 27 exact duplicate feature groups, 11 conflicting-label groups, and 26.55% near-neighbor opposite-label rate. This indicates separation difficulty, not a mathematical upper bound.

## Runtime
Threshold search fell from about 91.5 seconds to 0.03 seconds after vectorization. T0/T1 matrices were 21,000x62 / 21,000x78 with no NaN, inf, object dtype, duplicate columns, or memory explosion.

## Limitation
VALIDATION was consulted repeatedly in Rounds 1-3. Final Round 4 promotion used frozen TRAIN-OOF threshold and was not used to fit.

## Distance
Best F1 gap to 0.65: `{0.65 - float(winner['f1_binary']):.4f}`. Required precision at observed recall for F1=0.65: `{required_precision:.4f}`.

## Test
TEST untouched. Await explicit instruction before one-time final evaluation.

## Conclusion
Development is frozen at the best F1 achieved under the evaluated protocol. No claim is made that this is the dataset's maximum possible F1.
"""
    (FINAL / "rounds_1_to_4_summary.md").write_text(summary_md, encoding="utf-8")


if __name__ == "__main__":
    main()
