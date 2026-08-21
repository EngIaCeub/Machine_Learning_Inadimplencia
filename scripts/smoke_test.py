"""Minimal import/config smoke test for architecture and modeling contracts."""

from credit_default.config import get_project_config
from credit_default.features.credit_default_features import SEMANTIC_COLUMN_MAP
from credit_default.modeling.baseline import build_dummy_classifier
from credit_default.modeling.evaluate import REQUIRED_METRIC_GATES
from credit_default.modeling.optimize_f1 import ARTIFACT_DIR
from credit_default.modeling.train import build_candidate_estimators, build_validation_variant_specs
from credit_default.modeling.tune import COMPACT_SEARCH_SPACES


def main() -> None:
    cfg = get_project_config()
    assert cfg.dataset_id == 350
    assert REQUIRED_METRIC_GATES["roc_auc"] == 0.75
    assert build_dummy_classifier().strategy == "prior"
    assert "random_forest" in build_candidate_estimators()
    assert "logistic_regression" in COMPACT_SEARCH_SPACES
    assert "hist_gradient_boosting" in COMPACT_SEARCH_SPACES
    assert "xgboost" in COMPACT_SEARCH_SPACES
    assert SEMANTIC_COLUMN_MAP["X6"] == "PAY_0"
    assert any(spec.family_name == "random_forest" for spec in build_validation_variant_specs())
    assert any(spec.family_name == "hist_gradient_boosting" for spec in build_validation_variant_specs())
    assert any(spec.family_name == "xgboost" for spec in build_validation_variant_specs())
    assert ARTIFACT_DIR.name == "experiments"
    print("PASS: package imports and core contracts are valid.")


if __name__ == "__main__":
    main()
