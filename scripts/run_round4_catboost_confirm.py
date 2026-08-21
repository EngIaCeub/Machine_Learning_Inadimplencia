"""Promote only the selected CatBoost configuration to VALIDATION."""

from __future__ import annotations

import json

from credit_default.data.load import load_uci_dataset
from credit_default.data.split import make_splits
from credit_default.modeling.evaluate import evaluate_binary_classifier
from credit_default.modeling.round2 import ARTIFACT_DIR, _write_json
from credit_default.modeling.round4 import _cat_model, _catboost_frame


def main() -> None:
    features, target = load_uci_dataset()
    splits = make_splits(features, target)
    rows = __import__("pandas").read_csv(ARTIFACT_DIR / "round4_catboost_oof_experiments.csv")
    winner = rows.sort_values(["oof_f1", "oof_precision", "oof_auc"], ascending=False).iloc[0]
    params = json.loads(winner["hyperparameters"])
    threshold = float(winner["oof_threshold"])
    train, categorical = _catboost_frame(splits.X_train)
    validation, _ = _catboost_frame(splits.X_validation)
    model = _cat_model(params).fit(train, splits.y_train, cat_features=categorical)
    scores = model.predict_proba(validation)[:, 1]
    metrics = evaluate_binary_classifier(splits.y_validation, scores, threshold)
    result = {
        "experiment_id": "C1_frozen_validation",
        "model": "CatBoostClassifier",
        "feature_set": "A3",
        "oof": winner.to_dict(),
        "validation": metrics,
        "hyperparameters": params,
        "threshold_source": "TRAIN_OOF",
        "test_access": "none",
    }
    _write_json(ARTIFACT_DIR / "round4_catboost_validation_frozen.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
