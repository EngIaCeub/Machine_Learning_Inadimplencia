"""Run Round 3 segmented-threshold and TRAIN separability diagnostics."""

import json

from credit_default.data.load import load_uci_dataset
from credit_default.data.split import make_splits
from credit_default.modeling.round3 import run_round3_diagnostics


def main() -> None:
    features, target = load_uci_dataset()
    result = run_round3_diagnostics(make_splits(features, target))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
