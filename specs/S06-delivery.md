# S06 — Delivery and Reproducibility

Final completion requires all applicable checks to pass.

## Colab

- [ ] Fresh environment executes top-to-bottom.
- [ ] Dataset downloads automatically.
- [ ] No absolute local paths.
- [ ] No manual data upload.
- [ ] No credentials required.
- [ ] Seeds/config are visible.
- [ ] Metrics are regenerated.
- [ ] Charts/tables are regenerated.
- [ ] SHAP outputs regenerate.
- [ ] Demo regenerates.
- [ ] Results are compatible with report values.
- [ ] Exact successful dependency versions are frozen in `requirements.lock.txt`.

## Artifacts

- [ ] Executed Colab notebook exists.
- [ ] Notebook PDF export exists.
- [ ] Report exists.
- [ ] Final single PDF contains report + notebook export.
- [ ] Colab view link is populated and accessible.
- [ ] Video link is populated and accessible.
- [ ] No TODO/PLACEHOLDER remains in final deliverables.

## Required final verification

Run:
- unit/integration tests
- lint/static checks
- `scripts/smoke_test.py`
- `scripts/verify_artifacts.py`
- a clean Colab `Run all`

Artifact verification cannot override ML metric gates.
