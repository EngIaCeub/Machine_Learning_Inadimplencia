# S04 — Explainability and Demo

## Explainability

Champion model must have:

- global SHAP analysis
- at least one local SHAP explanation
- feature-importance view when model supports it
- concise business interpretation of major drivers

Do not claim causal effects. Describe model associations/influence.

## Demonstration

Create at least one new input with the exact inference schema.
Synthetic values may use Faker or controlled valid values.

Demo must show:

- input summary
- predicted class
- predicted probability
- threshold
- concise explanation of main feature contributions

The demo must call the same reusable prediction pipeline used by the project, not a notebook-only duplicate.
