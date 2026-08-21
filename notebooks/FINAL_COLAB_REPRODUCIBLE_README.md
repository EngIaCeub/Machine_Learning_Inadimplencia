# Final Colab Reproducible

## Purpose

`FINAL_COLAB_REPRODUCIBLE.ipynb` reproduces the frozen A1 solution from dataset loading through final evaluation. It is a `REPRODUCTION_ONLY` notebook: it does not tune, select models, search thresholds, alter features, or write official final artifacts.

## Frozen solution

- Model: `CatBoostClassifier`
- Features: A3 = `ROUND1 + BILL + PAYMENT`
- Positive class: `default = 1`
- Threshold: `0.247743`
- Official F1: Macro F1
- Expected TEST gates: AUC `0.7865`, Recall `0.6104`, Macro F1 `0.7019`, `3/3 PASS`

## Colab use

1. Open `notebooks/FINAL_COLAB_REPRODUCIBLE.ipynb` in Google Colab.
2. Select a CPU runtime and choose **Runtime > Run all**.
3. Save the executed notebook after confirming the reproducibility table.

`REPO_URL` is preconfigured as `https://github.com/EngIaCeub/Machine_Learning_Inadimplencia.git`. The Colab publication URL is `https://colab.research.google.com/drive/1_Jgx1HXY3wPmAU8GJB8B347WKp0kumSV`. No manual upload, personal Drive, Windows path, credential, or GPU is required.

## Dependencies

The notebook installs `requirements.txt` and pins `catboost==1.2.10` for the final model. It uses the official UCI loader, pandas, NumPy, scikit-learn, matplotlib, and CatBoost. GPU is not required.

## Reproducibility notes

The notebook loads `development_winner.json` for frozen configuration, reads the persisted Round 4 split manifest, reconstructs A3 with project functions, fits only on TRAIN, and calculates Validation/Test metrics in memory. It does not read `final_test_predictions.csv` or `final_test_metrics.json` as prediction sources and does not overwrite official artifacts.

The historical TEST physical access count is 2 because the original final evaluator required one deterministic persistence recovery. That event is documented in `artifacts/final/leakage_audit.json`; this notebook is not a development run and does not change the final candidate.
