"""Run the validation-only binary-F1 optimization workflow."""

from __future__ import annotations

import json

from credit_default.data.load import load_uci_dataset
from credit_default.modeling.optimize_f1 import run_binary_f1_optimization_from_dataset


def main() -> None:
    features, target = load_uci_dataset()
    result = run_binary_f1_optimization_from_dataset(features, target)
    payload = {
        "artifact_dir": str(result.artifact_dir),
        "baseline": {
            "model": result.baseline.model,
            "threshold": result.baseline.threshold,
            "roc_auc": result.baseline.roc_auc,
            "precision": result.baseline.precision,
            "recall": result.baseline.recall,
            "f1_binary": result.baseline.f1_binary,
            "macro_f1": result.baseline.macro_f1,
            "weighted_f1": result.baseline.weighted_f1,
        },
        "best_result": {
            "model": result.best_result.model,
            "strategy": result.best_result.strategy,
            "threshold": result.best_result.threshold,
            "roc_auc": result.best_result.roc_auc,
            "precision": result.best_result.precision,
            "recall": result.best_result.recall,
            "f1_binary": result.best_result.f1_binary,
            "macro_f1": result.best_result.macro_f1,
            "weighted_f1": result.best_result.weighted_f1,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
