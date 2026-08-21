---
name: plan-spec
description: Plan architecture or convert a new/changed requirement into a compact executable spec. Use for cross-module design; do not use for routine implementation.
---

1. Read `specs/INDEX.md`.
2. Read only impacted specs/files.
3. Identify decision, scope, dependencies, acceptance criteria, tests, risks.
4. Prefer the smallest design that satisfies S00/S01.
5. If work spans multiple modules or changes contracts, write a compact plan under `specs/exec/`.
6. Do not write production implementation code.

Return only: STATUS, decision, files/scope, acceptance, risks.
