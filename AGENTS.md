# Mission

Build a reproducible academic ML project for Track A / UCI Credit Default.
The source of truth is `specs/`. Keep the notebook thin; reusable logic belongs in `src/`.

## Caveman mode

- Be terse. Do not narrate routine tool calls.
- Do not repeat the user request or dump full logs.
- Report only: `STATUS`, changed files, checks, risks/blockers.
- Prefer `path:line`, metric values, and short bullets over prose.
- Never expose hidden reasoning. Return conclusions and evidence only.
- `DONE` means relevant checks were actually executed.

Default completion format:

```text
STATUS: DONE|BLOCKED|NEEDS_REVIEW
changed:
- path
checks:
- command/result
risk:
- none|short note
```

## Context economy

1. Read `specs/INDEX.md`.
2. Open only the spec(s) relevant to the task.
3. Load only the skill(s) needed for the task.
4. Read only files required to make or verify the change.
5. Do not recursively scan `docs/`, `artifacts/`, notebooks, or the entire repository unless required.
6. Summarize subagent results; do not paste their full transcripts.
7. Use `STATE.md` for current project status instead of reconstructing history.

## Delegation

- Architecture, cross-module design, spec creation/change -> `architect`.
- ML methodology, leakage, metric validity, threshold/tuning concerns -> `ml_reviewer`.
- Python implementation and refactoring inside an approved spec -> `implementer`.
- Tests, lint, smoke checks, reproducibility checks -> `verifier`.
- File existence, links, exports, PDF/notebook/artifact QA -> `artifact_ops`.

Escalation:
`artifact_ops/verifier -> implementer -> ml_reviewer -> architect`

Do not start with the most expensive agent unless the task genuinely needs architecture or difficult methodological judgment.

## Implementation rules

- Prefer simple functions and explicit dependencies.
- No business/model logic hidden in notebook cells.
- No absolute local paths, credentials, manual dataset upload, or hidden global state.
- Fix deterministic seeds where supported.
- Any transformer that learns parameters must be fit on training data only.
- Preserve train/validation/test isolation.
- Significant behavior changes require acceptance criteria before implementation.
- Changed behavior requires tests.
- Never weaken an academic metric gate to make a test pass.

## Project gates

Read `specs/S00-project-contract.md` for the academic contract.
Never claim final completion before `specs/S06-delivery.md` is satisfied.
