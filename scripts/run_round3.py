"""Run Round 3 validation-only hard-negative and cascade experiments."""

import json

from credit_default.data.load import load_uci_dataset
from credit_default.modeling.round3 import run_round3_from_dataset


def main() -> None:
    features, target = load_uci_dataset()
    result = run_round3_from_dataset(features, target)
    print(json.dumps({"baseline": result["baseline"], "best": result["best"], "artifact_dir": str(result["artifact_dir"])}, indent=2, default=str))


if __name__ == "__main__":
    main()
