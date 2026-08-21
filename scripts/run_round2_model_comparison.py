"""Run Round 2 controlled model comparison on validation only."""

import json

from credit_default.data.load import load_uci_dataset
from credit_default.data.split import make_splits
from credit_default.modeling.round2 import run_round2_model_comparison


def main() -> None:
    features, target = load_uci_dataset()
    result = run_round2_model_comparison(make_splits(features, target))
    print(json.dumps({"best": result["best"], "diversity": result["diversity"], "best_ensemble": result["best_ensemble"], "artifact_dir": str(result["artifact_dir"])}, indent=2, default=str))


if __name__ == "__main__":
    main()
