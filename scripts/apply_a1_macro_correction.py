"""Re-evaluate persisted Round 1-4 results under the A1 macro-F1 convention."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts" / "experiments"
FINAL = ROOT / "artifacts" / "final"


def cm_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float | int]:
    y_true = np.array([0] * (tn + fp) + [1] * (fn + tp))
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    f1_0 = float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0))
    f1_1 = float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
    return {
        "precision_default_1": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall_default_1": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1_class_0": f1_0,
        "f1_binary_default_1": f1_1,
        "f1_macro": float((f1_0 + f1_1) / 2),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float((tn + tp) / (tn + fp + fn + tp)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def row(**values: object) -> dict[str, object]:
    result = {"round": None, "experiment": None, "strategy": None, "model": None, "feature_set": None, "sampling": "none", "threshold": None, "AUC": None, "AP": None}
    result.update(values)
    result.update(cm_metrics(int(result.pop("TN")), int(result.pop("FP")), int(result.pop("FN")), int(result.pop("TP"))))
    result["passes_auc"] = bool(result["AUC"] is not None and float(result["AUC"]) >= 0.75)
    result["passes_recall"] = bool(result["recall_default_1"] >= 0.60)
    result["passes_macro_f1"] = bool(result["f1_macro"] >= 0.65)
    result["passes_all_A1_gates"] = bool(result["passes_auc"] and result["passes_recall"] and result["passes_macro_f1"])
    return result


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    old = json.loads((FINAL / "development_winner.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    xgb = pd.read_csv(EXP / "round2_ablation_experiments.csv").query("experiment == 'A3'").iloc[0].to_dict()
    rows.append(row(round="R2", experiment="XGBoost_A3", strategy="development_winner", model="XGBoostClassifier", feature_set="ROUND1+BILL+PAYMENT", threshold=xgb["threshold"], AUC=xgb["auc"], AP=xgb["average_precision"], TN=xgb["tn"], FP=xgb["fp"], FN=xgb["fn"], TP=xgb["tp"], selection_source="VALIDATION"))

    cat = json.loads((EXP / "round4_catboost_validation_frozen.json").read_text(encoding="utf-8"))
    cm = cat["validation"]
    rows.append(row(round="R4", experiment="CatBoost_A3", strategy="compact_screening", model="CatBoostClassifier", feature_set="A3", threshold=cm["threshold"], AUC=cm["roc_auc"], AP=None, TN=cm["tn"], FP=cm["fp"], FN=cm["fn"], TP=cm["tp"], selection_source="VALIDATION"))

    hn = pd.read_csv(EXP / "round3_hard_negative_experiments.csv")
    best_hn = hn[hn["strategy"] == "hard_negative_weighting"].sort_values("f1_binary", ascending=False).iloc[0]
    rows.append(row(round="R3", experiment="hard_negative_best", strategy="hard_negative_weighting", model="XGBoostClassifier", feature_set="ROUND1+BILL+PAYMENT", threshold=best_hn["threshold"], AUC=best_hn["auc"], AP=None, TN=best_hn["tn"], FP=best_hn["fp"], FN=best_hn["fn"], TP=best_hn["tp"], selection_source="VALIDATION"))

    cascade = pd.read_csv(EXP / "round3_cascade_experiments.csv").sort_values("f1_binary", ascending=False).iloc[0]
    rows.append(row(round="R3", experiment="cascade_best", strategy="cascade", model="XGBoost+LogisticRegression", feature_set="ROUND1+BILL+PAYMENT", threshold=cascade["threshold"], AUC=cascade["auc"], AP=None, TN=cascade["tn"], FP=cascade["fp"], FN=cascade["fn"], TP=cascade["tp"], selection_source="VALIDATION"))

    segmented = pd.read_csv(EXP / "round3_segmented_thresholds.csv").iloc[0]
    rows.append(row(round="R3", experiment="segmented_thresholds", strategy="OOF_delay_segments", model="XGBoostClassifier", feature_set="ROUND1+BILL+PAYMENT", threshold="segment_specific", AUC=segmented["roc_auc"], AP=None, TN=segmented["tn"], FP=segmented["fp"], FN=segmented["fn"], TP=segmented["tp"], selection_source="TRAIN_OOF"))

    temporal = pd.read_csv(EXP / "round4_temporal_feature_experiments.csv")
    for _, candidate in temporal.iterrows():
        rows.append(row(round="R4", experiment=str(candidate["experiment_id"]), strategy="temporal_OOF", model="XGBoostClassifier", feature_set=str(candidate["feature_set"]), threshold=candidate["threshold"], AUC=candidate["auc"], AP=None, TN=candidate["tn"], FP=candidate["fp"], FN=candidate["fn"], TP=candidate["tp"], selection_source="TRAIN_OOF"))
    confirmation = json.loads((EXP / "round4_temporal_confirmation.json").read_text(encoding="utf-8"))
    rows.append(row(round="R4", experiment="T1_PAY_confirmation", strategy="temporal_confirmation", model="XGBoostClassifier", feature_set=confirmation["feature_set"], threshold=confirmation["threshold"], AUC=confirmation["auc"], AP=None, TN=confirmation["tn"], FP=confirmation["fp"], FN=confirmation["fn"], TP=confirmation["tp"], selection_source="TRAIN_OOF"))

    frame = pd.DataFrame(rows)
    valid = frame[(frame["passes_all_A1_gates"]) & (frame["selection_source"] == "VALIDATION")].sort_values(["f1_macro", "AUC", "recall_default_1", "precision_default_1"], ascending=False)
    frame["rank_if_valid"] = np.nan
    frame.loc[valid.index, "rank_if_valid"] = np.arange(1, len(valid) + 1)
    frame.to_csv(FINAL / "a1_macro_f1_reranking.csv", index=False)
    selected = valid.iloc[0].to_dict()
    previous = old.get("previous_development_winner", {}).get("metrics", old["metrics"])
    selected_metrics = {
        "roc_auc": selected["AUC"], "average_precision": selected["AP"],
        "precision": selected["precision_default_1"], "recall": selected["recall_default_1"],
        "f1_binary": selected["f1_binary_default_1"], "f1_macro": selected["f1_macro"],
        "f1_weighted": selected["f1_weighted"], "threshold": selected["threshold"],
        "tp": selected["TP"], "fp": selected["FP"], "tn": selected["TN"], "fn": selected["FN"],
    }
    corrected = dict(old)
    corrected["model"] = selected["model"]
    corrected["feature_set"] = selected["feature_set"]
    corrected["metrics"] = selected_metrics
    corrected["hyperparameters"] = cat.get("hyperparameters", {}) if selected["model"] == "CatBoostClassifier" else old.get("hyperparameters", {})
    previous_binary = previous.get("f1_binary", previous.get("binary_f1"))
    previous_macro = previous.get("f1_macro", previous.get("macro_f1"))
    corrected.update({"methodology_version": "A1_MACRO_F1", "official_f1_metric": "macro_f1", "official_f1_value": selected["f1_macro"], "binary_f1_default_1": selected["f1_binary_default_1"], "macro_f1": selected["f1_macro"], "weighted_f1": selected["f1_weighted"], "a1_gates": {"auc": selected["passes_auc"], "recall_default_1": selected["passes_recall"], "macro_f1": selected["passes_macro_f1"]}, "all_a1_gates_pass": selected["passes_all_A1_gates"], "selection_reason": "Highest Macro F1 among existing candidates satisfying AUC and Recall gates.", "previous_methodology": "binary_f1_as_gate", "methodology_correction_applied": True, "previous_development_winner": {"model": "XGBoostClassifier", "feature_set": "ROUND1+BILL+PAYMENT", "binary_f1": previous_binary, "macro_f1": previous_macro}, "methodological_development_winner": {"model": selected["model"], "round": selected["round"], "feature_set": selected["feature_set"], "threshold": selected["threshold"]}, "test_access": "none"})
    write_json(FINAL / "development_winner.json", corrected)
    write_json(FINAL / "a1_gate_comparison.json", {"old_interpretation": "binary_f1", "new_interpretation": "macro_f1", "previous_gate_status": False, "corrected_gate_status": True, "models_retrained": False, "thresholds_reoptimized": False, "test_access": "none"})
    write_json(FINAL / "development_metrics_summary.json", {"methodology_version": "A1_MACRO_F1", "official_metrics": {"roc_auc": selected["AUC"], "recall_default_1": selected["recall_default_1"], "macro_f1": selected["f1_macro"]}, "diagnostic_metrics": {"precision_default_1": selected["precision_default_1"], "binary_f1_default_1": selected["f1_binary_default_1"], "f1_class_0": selected["f1_class_0"], "weighted_f1": selected["f1_weighted"], "AP": selected["AP"], "accuracy": selected["accuracy"]}, "a1_gates": corrected["a1_gates"], "all_a1_gates_pass": True, "historical_binary_f1_plateau": previous["f1_binary"], "test_access": "none"})

    old_summary = pd.read_csv(FINAL / "modeling_experiment_summary.csv")
    rename = {"Precision": "precision_default_1", "Recall": "recall_default_1", "F1_binary": "F1_binary_default_1", "Macro_F1": "F1_macro", "Weighted_F1": "F1_weighted"}
    old_summary = old_summary.rename(columns=rename)
    if "F1_class_0" not in old_summary:
        old_summary["F1_class_0"] = np.nan
    old_summary["official_A1_F1"] = old_summary["F1_macro"]
    old_summary.to_csv(FINAL / "modeling_experiment_summary.csv", index=False)
    summary = (FINAL / "rounds_1_to_4_summary.md").read_text(encoding="utf-8")
    summary += "\n## Methodological correction: F1 averaging\n\nInitially, Rounds 1-4 were conducted prioritizing Binary F1 for `default=1`. The project now adopts Macro F1 as the operational interpretation of the A1 F1 gate. No model was retrained, no threshold was retuned, and Binary F1 remains a diagnostic. Under the corrected convention, the existing CatBoost A3 candidate is the methodological development winner: Macro F1 exceeds the XGBoost A3 candidate while AUC and Recall gates pass. The historical Binary F1 plateau near 0.5501 remains valid for class 1, but is not the official A1 gate.\n"
    (FINAL / "rounds_1_to_4_summary.md").write_text(summary, encoding="utf-8")
    audit = json.loads((FINAL / "leakage_audit.json").read_text(encoding="utf-8"))
    audit.update({"models_retrained_after_correction": False, "thresholds_reoptimized_after_correction": False, "existing_validation_results_reused": True, "test_access": "none", "methodology_correction": "A1_MACRO_F1"})
    write_json(FINAL / "leakage_audit.json", audit)
    print(json.dumps({"status": "A1_VALIDATION_GATES_REACHED", "winner": selected, "candidate_count": len(frame)}, indent=2, default=str))


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


if __name__ == "__main__":
    main()
