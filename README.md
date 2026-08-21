# ML Credit Default — Final Academic Delivery

Reproducible **Track A / UCI Default of Credit Card Clients** academic project.

The project contains the approved data pipeline, validation-only development evidence, a frozen final candidate, and the A1 holdout evaluation.

## Final status

`FINAL_ACADEMIC_DELIVERY_READY`

Final model: `CatBoostClassifier`, feature set A3 (`ROUND1 + BILL + PAYMENT`), threshold `0.247743`.

Final TEST gates: ROC-AUC `0.7865` PASS, Recall `0.6104` PASS, Macro F1 `0.7019` PASS (`3/3`). `default=1` is the positive class. Binary F1 remains a diagnostic at approximately `0.55` and is not the official A1 gate.

## Start here

1. Read `AGENTS.md`.
2. Read `specs/INDEX.md`.
3. Inspect `STATE.md`.
4. Implement one approved spec at a time.
5. Run `python scripts/verify_architecture.py`.

## Main folders

- `.codex/agents/` — project-scoped custom agents.
- `.agents/skills/` — repository skills discovered by Codex.
- `specs/` — compact source-of-truth contracts.
- `src/credit_default/` — clean ML application modules.
- `notebooks/` — thin Colab interface only.
- `tests/` — automated verification.
- `scripts/` — reproducibility and artifact checks.
- `artifacts/` — generated outputs, not source code.

See `README_ARCHITECTURE.md` for model routing and operating rules.

## Documentation

- `docs/RELATORIO_FINAL.md` — complete technical report.
- `docs/RESUMO_EXECUTIVO.md` — short submission summary.
- `docs/GUIA_REPRODUCAO.md` — reproducibility and artifact guide.
- `delivery/DELIVERY_MANIFEST.md` — final package manifest.

## Final artifacts

Evidence is persisted under `artifacts/final/`, including the frozen winner, validation/test comparison, final metrics, gate status, leakage audit, predictions, and plots. The TEST holdout has already been consumed and must not be rerun during normal reproduction.

## Install and verify

```powershell
python -m pip install -r requirements.txt
python -m pytest
python scripts/smoke_test.py
python scripts/verify_artifacts.py
```
