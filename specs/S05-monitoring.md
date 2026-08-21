# S05 — Monitoring

## Production proposal

Monitor four layers:

1. Data quality.
2. Data drift.
3. Prediction drift.
4. Realized model performance once labels arrive.

## Minimum indicators

Data:
- missing rate
- invalid-range rate
- schema failures
- feature distributions

Drift:
- PSI or equivalent distribution comparison
- material shift in high-importance features

Prediction:
- positive prediction rate
- score/probability distribution

Performance:
- ROC-AUC
- F1
- Recall
- Precision
- confusion matrix trend

## Retraining

Define:
- periodic review cadence
- event-based triggers
- minimum new labeled-data requirement
- shadow/challenger validation before replacement
- rollback/version retention

Thresholds used in the report must be presented as proposed operational policy, not fabricated empirical facts unless calculated from project data.
