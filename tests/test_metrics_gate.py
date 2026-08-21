import pandas as pd

from credit_default.modeling.evaluate import evaluate_binary_classifier, passes_academic_gates


def test_academic_gate_passes_at_exact_thresholds():
    metrics = {"roc_auc": 0.75, "f1": 0.65, "recall": 0.60}
    assert passes_academic_gates(metrics)


def test_academic_gate_fails_below_any_required_metric():
    metrics = {"roc_auc": 0.80, "f1": 0.64, "recall": 0.70}
    assert not passes_academic_gates(metrics)


def test_a1_uses_macro_f1_as_official_metric():
    y_true = pd.Series([0] * 3505 + [1] * 995)
    y_pred = [0] * 2900 + [1] * 605 + [0] * 388 + [1] * 607
    scores = pd.Series(y_pred, dtype=float).to_numpy()
    metrics = evaluate_binary_classifier(y_true, scores, threshold=0.5)

    assert metrics["f1_binary_default_1"] == metrics["f1_binary"]
    assert abs(metrics["f1_macro"] - 0.7019) < 0.001
    assert metrics["official_a1_f1"] == metrics["f1_macro"]
    assert passes_academic_gates({"roc_auc": 0.7817, "recall_positive": metrics["recall_positive"], "f1_macro": metrics["f1_macro"]})
