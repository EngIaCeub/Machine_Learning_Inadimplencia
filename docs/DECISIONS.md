# Compact Architecture Decisions

## ADR-001
Decision: use UCI dataset id 350 via `ucimlrepo`.
Why: public, no local upload/authentication required.
Impact: S00, S02, S06.

## ADR-002
Decision: lightweight clean architecture; notebook is orchestration only.
Why: clarity, testability, reproducibility without unnecessary abstraction.
Impact: S01.

## ADR-003
Decision: model ladder = Dummy -> Logistic Regression -> Decision Tree -> Random Forest.
Why: demonstrates baseline, linear, simple nonlinear, and ensemble progression.
Impact: S03.

## ADR-004
Decision: test set is untouched until champion and threshold are frozen.
Why: honest final evaluation.
Impact: S02, S03.

## ADR-005
Decision: Sol/High for architecture/methodology, Terra/Medium for implementation, Luna for verification/artifacts.
Why: allocate reasoning budget to high-cost errors while keeping routine work economical.
Impact: `.codex/agents/`.

## ADR-006
Decision: repository skills live in `.agents/skills/`.
Why: current Codex repository skill discovery path.
Impact: skills loading/context economy.

## ADR-007
Decision: amend S03 to add `HistGradientBoostingClassifier` after Random Forest as an additional validation-only candidate.
Why: validation-only evidence showed the LR/DT/RF ladder plateaued well below the required F1 gate even after semantic fixes, feature engineering, imbalance handling, threshold search, and compact tuning.
Impact: `specs/S03-modeling.md`, validation-only S03 experimentation, future final-test candidate set.

## ADR-008
Decision: amend S03-v3 to add `XGBoostClassifier` after `HistGradientBoostingClassifier` as one additional validation-only candidate.
Why: RF and HGB converged near the same validation F1 ceiling, so one stronger tabular boosting family is justified before reconsidering the gate or broader methodology.
Impact: `specs/S03-modeling.md`, `requirements.txt`, validation-only S03 experimentation.
