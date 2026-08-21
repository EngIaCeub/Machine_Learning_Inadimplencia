# S00 — Project Contract

## Fixed scope

- Track: A — supervised binary classification.
- Problem: predict next-month credit-card default.
- Dataset: UCI Default of Credit Card Clients, dataset id 350.
- Public ingestion target: `ucimlrepo.fetch_ucirepo(id=350)`.
- Explainability: SHAP plus model-native feature importance when available.

## Required academic gates

Final untouched test set must satisfy:

- ROC-AUC >= 0.75
- F1 >= 0.65
- Recall >= 0.60

## Required lifecycle

- ingestion
- EDA
- cleaning/preparation
- feature engineering
- encoding/scaling when appropriate
- train/validation/test separation
- baseline
- multiple candidate models
- hyperparameter tuning
- quantitative evaluation
- explainability
- prediction on new/synthetic data
- monitoring/drift proposal
- retraining strategy
- reproducible Colab
- final report + executed notebook + Colab link + video link

## Non-negotiable rules

- No weakening metric gates.
- No test-set-driven tuning.
- No leakage.
- No local absolute paths.
- No manual dataset upload for core execution.
- No credentials in source.
