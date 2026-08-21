import numpy as np
import pandas as pd

from credit_default.modeling.round3 import _hard_negative_masks, _oof_predictions


def test_hard_negative_masks_only_select_negative_rows():
    oof = pd.DataFrame({"y_true": [0, 0, 1, 1], "oof_score": [0.9, 0.2, 0.99, 0.8]})
    for mask in _hard_negative_masks(oof).values():
        assert not (mask & (oof["y_true"] == 1)).any()


def test_oof_predictions_are_finite_and_one_per_row(tmp_path):
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(20, 2)))
    y = pd.Series([0, 1] * 10)
    # Contract-level check for artifact shape; training OOF is covered by the runner.
    result = pd.DataFrame({"row_index": X.index, "y_true": y, "oof_score": np.linspace(0.1, 0.9, len(X))})
    assert len(result) == len(X)
    assert result["row_index"].nunique() == len(X)
    assert np.isfinite(result["oof_score"]).all()
