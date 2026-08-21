"""Run Round 4 CatBoost and temporal validation-only experiments."""

import json

from credit_default.data.load import load_uci_dataset
from credit_default.modeling.round4 import run_round4_from_dataset


def main() -> None:
    features, target = load_uci_dataset()
    result = run_round4_from_dataset(features, target)
    print(json.dumps({"best": result["best"], "comparison": result["comparison"].to_dict(orient="records"), "artifact_dir": str(result["artifact_dir"])}, indent=2, default=str))


if __name__ == "__main__":
    main()
