# S01 — Architecture

## Boundary

Use lightweight clean architecture, not ceremonial enterprise layering.

```text
data -> features -> modeling -> explainability -> operations
```

Notebook is an orchestration/view layer.

## Modules

- `data/load.py`: acquisition only.
- `data/schema.py`: schema/target validation.
- `data/split.py`: deterministic stratified split.
- `features/preprocessing.py`: sklearn-compatible preprocessing/features.
- `modeling/baseline.py`: DummyClassifier baseline.
- `modeling/train.py`: candidate training.
- `modeling/tune.py`: validation-only hyperparameter search.
- `modeling/evaluate.py`: metrics/plots/data structures.
- `explainability/shap_analysis.py`: global/local explanations.
- `operations/predict.py`: reusable prediction interface.
- `operations/monitoring.py`: monitoring calculations/proposal helpers.

## Code rules

- Prefer pure functions.
- Avoid hidden mutable globals.
- Configuration lives in `config.py`.
- Any fitted transform learns from train only.
- Keep plotting separate from model-state mutation.
- Keep notebook cells thin and readable.
- Tests mirror behavioral boundaries.

## Reproducibility

- Central `RANDOM_SEED`.
- Explicit deterministic splits.
- Stable package lock after first green fresh run.
- Dataset acquisition automated.
- Generated outputs written under `artifacts/`.

## Forbidden

- absolute local paths
- hardcoded secrets
- manual core-data upload
- fitting scaler/encoder before split
- using test set for hyperparameter or threshold selection
- duplicating core logic in notebook
