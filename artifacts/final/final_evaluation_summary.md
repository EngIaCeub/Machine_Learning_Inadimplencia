# Final Evaluation

Frozen candidate: CatBoost A3. Official F1: Macro F1. Positive class: `default=1`. Threshold: `0.247743`.

Validation: AUC `0.7811`, Recall `0.6050`, Macro F1 `0.7025`.

TEST: AUC `0.7865`, AP `0.5639`, Precision `0.5008`, Recall `0.6104`, Binary F1 `0.5502`, Macro F1 `0.7019`, Weighted F1 `0.7865`.

Gates: AUC `True`, Recall `True`, Macro F1 `True`. Overall: `True`.

The final TEST set was evaluated only after model, feature set, threshold and metric definition had been frozen. No model selection, threshold optimization or hyperparameter tuning was performed using TEST results.
