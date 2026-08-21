import numpy as np
import pandas as pd

from credit_default.modeling.evaluate import build_threshold_search_table, select_best_threshold_row
from credit_default.modeling.round4 import _fast_threshold


def test_fast_threshold_matches_official_threshold_selection():
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    scores = np.array([0.08, 0.72, 0.31, 0.68, 0.22, 0.54, 0.41, 0.12, 0.49, 0.03])

    expected = float(select_best_threshold_row(build_threshold_search_table(y, scores))["threshold"])

    assert _fast_threshold(y, scores) == expected
