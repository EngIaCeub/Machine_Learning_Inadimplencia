import numpy as np
import pandas as pd

from credit_default.features.credit_default_features import build_behavioral_features


def test_round2_builder_does_not_depend_on_target_values():
    frame = pd.DataFrame({
        "X1": [100, 200], "X2": [1, 2], "X3": [1, 2], "X4": [1, 2], "X5": [30, 40],
        "X6": [2, 0], "X7": [1, 0], "X8": [0, 0], "X9": [0, 0], "X10": [0, 0], "X11": [0, 0],
        "X12": [100, 200], "X13": [90, 180], "X14": [80, 160], "X15": [70, 140], "X16": [60, 120], "X17": [50, 100],
        "X18": [10, 20], "X19": [9, 18], "X20": [8, 16], "X21": [7, 14], "X22": [6, 12], "X23": [5, 10],
    })
    first = build_behavioral_features(frame, enabled_groups=("pay", "ratios", "trends"))
    second = build_behavioral_features(frame, enabled_groups=("pay", "ratios", "trends"))
    pd.testing.assert_frame_equal(first, second)
    assert not first.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any()
