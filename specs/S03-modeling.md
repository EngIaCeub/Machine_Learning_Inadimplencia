# S03 — Modeling

## Required model ladder

1. `DummyClassifier` baseline.
2. Logistic Regression.
3. Decision Tree.
4. Random Forest.
5. HistGradientBoostingClassifier.
6. XGBoostClassifier.

`HistGradientBoostingClassifier` is an additional candidate, not an automatic champion.
`XGBoostClassifier` is an additional candidate, not an automatic champion.
Further models still require explicit justification; do not add complexity by default.

## Tuning

- Tune only after baseline candidate evaluation.
- Validation data or CV internal to training may guide hyperparameters.
- Test data remains untouched until champion selection is frozen.
- Search space must be compact and documented.
- Optimize for the academic objective without hiding trade-offs among ROC-AUC, F1 and Recall.
- Threshold tuning, if used, is selected on validation data only and stored explicitly.
- For `HistGradientBoostingClassifier`, compact tuning may cover:
  - `learning_rate`
  - `max_iter`
  - `max_leaf_nodes`
  - `max_depth`
  - `min_samples_leaf`
  - `l2_regularization`
- For `XGBoostClassifier`, compact randomized tuning may cover:
  - `n_estimators`
  - `max_depth`
  - `learning_rate`
  - `min_child_weight`
  - `subsample`
  - `colsample_bytree`
  - `reg_lambda`
  - `reg_alpha`
  - `scale_pos_weight`
- Sample weighting for imbalance is allowed when methodologically justified.

## Validation-only amendment

- Preserve the current engineered feature pipeline and preprocessing contract.
- Candidate comparison remains validation-only until a frozen champion is ready for final test evaluation.
- Thresholds must not be reused across model families; each candidate stores its own validation threshold.
- `HistGradientBoostingClassifier` must expose `predict_proba` and be compared directly against Logistic Regression, Decision Tree, and Random Forest.
- `XGBoostClassifier` must expose `predict_proba` and be compared directly against Random Forest and HistGradientBoosting.
- `XGBoostClassifier` requires dependency `xgboost`.
- Default backend for `XGBoostClassifier` is:
  - `tree_method="hist"`
  - `device="cpu"`
- GPU is optional acceleration only:
  - `device="cuda"`
  - CPU fallback is mandatory
  - backend choice must not change split, preprocessing, threshold logic, champion rule, or acceptance gates

## Champion gate

Evaluate the frozen champion once on test.

Required:
- ROC-AUC >= 0.75
- F1 >= 0.65
- Recall >= 0.60

Also report:
- precision
- accuracy
- confusion matrix
- ROC curve
- threshold used

## Failure escalation

- implementation/test issue -> implementer
- unexpected metrics, leakage concern, threshold/tuning question -> ml_reviewer
- cross-module contract change -> architect
