---
name: model-experiment
description: Train, tune, compare, select, and evaluate Track A classification models under S03 without leaking the test set.
---

Read S00 and S03 plus only the model/data interfaces used.

Order:
1. Dummy baseline.
2. Logistic Regression.
3. Decision Tree.
4. Random Forest.
5. Compact tuning.
6. Freeze champion and threshold.
7. One final test evaluation.

Never tune on test.
Record ROC-AUC, F1, Recall, Precision, Accuracy, confusion matrix.
If required metrics fail for non-obvious reasons, stop and escalate to `ml_reviewer`.
