# Architecture Operating Guide

## Model routing

| Role | Model | Effort | Purpose |
|---|---|---|---|
| architect | `gpt-5.6-sol` | high | architecture, specs, difficult cross-cutting decisions |
| ml_reviewer | `gpt-5.6-sol` | high | leakage, metrics, experiment validity, ML judgment |
| implementer | `gpt-5.6-terra` | medium | coding, refactors, tests under approved specs |
| verifier | `gpt-5.6-luna` | medium | tests, static/repro checks, focused verification |
| artifact_ops | `gpt-5.6-luna` | low | files, links, exports, artifact existence/shape |

The root session defaults subagents to Terra/Medium. Custom agent files override that choice.

If a specific Codex surface does not expose one of these exact model IDs, change only the corresponding `.codex/agents/*.toml` model value; responsibilities and effort tiers remain the same.

## Context strategy

```text
AGENTS.md
  -> specs/INDEX.md
  -> relevant SPEC
  -> relevant SKILL
  -> relevant source/test files
```

Avoid loading the complete project history. `STATE.md` is the compact handoff.

## Clean architecture (lightweight)

```text
data -> features -> modeling -> explainability -> operations
                                     ^
                                     |
                                  notebook
```

The notebook orchestrates; it does not own reusable logic.

## Human gates

1. Architecture.
2. Dataset + EDA + preprocessing.
3. Champion model + metrics.
4. Final delivery.

## Done means

A task is only `DONE` after its relevant checks have executed and evidence is available.
