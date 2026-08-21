"""Run Round 2 feature experiments on train/validation only."""

import json

from credit_default.data.load import load_uci_dataset
from credit_default.modeling.round2 import run_round2_feature_experiments_from_dataset


def main() -> None:
    features, target = load_uci_dataset()
    result = run_round2_feature_experiments_from_dataset(features, target)
    print(json.dumps({"best": result["best"], "artifact_dir": str(result["artifact_dir"])}, indent=2, default=str))


if __name__ == "__main__":
    main()
