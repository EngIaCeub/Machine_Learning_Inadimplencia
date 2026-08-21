"""Run the executable Round 4 temporal branch on validation only."""

import json

from credit_default.data.load import load_uci_dataset
from credit_default.modeling.round4 import run_round4_temporal_only_from_dataset


def main() -> None:
    features, target = load_uci_dataset()
    result = run_round4_temporal_only_from_dataset(features, target)
    print(json.dumps({"best": result["best"], "temporal_oof": result["temporal_oof"].to_dict(orient="records"), "artifact_dir": str(result["artifact_dir"])}, indent=2, default=str))


if __name__ == "__main__":
    main()
