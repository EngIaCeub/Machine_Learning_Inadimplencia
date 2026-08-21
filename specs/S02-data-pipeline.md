# S02 — Data Pipeline

## Flow

```text
UCI id=350
 -> raw features/target
 -> schema validation
 -> data quality checks
 -> EDA
 -> train/validation/test
 -> fit preprocessing on train
 -> transform validation/test
```

## Acceptance criteria

- Dataset loads without authentication or local files.
- Expected order of magnitude: ~30,000 observations.
- Target is resolved explicitly and normalized to a 1D binary series.
- Duplicate/missing/type checks are reported.
- EDA covers target balance and relevant numeric/categorical distributions.
- Split is stratified and deterministic.
- Train/validation/test are distinct.
- Preprocessing is fit only on training data.
- Any feature engineering is documented and reproducible.
- No identifier is used as a predictive signal unless explicitly justified.

## Preferred split

Initial default: 70% train / 15% validation / 15% test, stratified.
Changing this requires an explicit methodological reason.

## Gate output

Data-stage summary must include:
- shapes
- class prevalence per split
- schema issues
- preprocessing columns/steps
- leakage checks
