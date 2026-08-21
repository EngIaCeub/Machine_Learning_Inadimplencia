---
name: data-pipeline
description: Implement or verify UCI data loading, schema checks, EDA helpers, deterministic splitting, preprocessing, and feature engineering for S02.
---

Read S00, S01, S02 only.

Rules:
- acquire dataset automatically
- keep target resolution explicit
- split before fitting learned transforms
- fit preprocessing on train only
- preserve split class prevalence checks
- avoid ID leakage
- expose reusable functions under `src/credit_default/`
- keep notebook orchestration-only

Add tests for split determinism, schema expectations, and preprocessing fit/transform behavior.
Escalate methodological ambiguity to `ml_reviewer`.
