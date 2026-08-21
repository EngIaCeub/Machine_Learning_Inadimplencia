---
name: repo-check
description: Run lightweight repository architecture, import, lint, and test checks after scaffold or focused code changes.
---

Run smallest checks first:
1. `python scripts/verify_architecture.py`
2. `python scripts/smoke_test.py`
3. relevant pytest target
4. broader `pytest`/`ruff` only when needed

Do not paste full logs. Report command + result + first actionable error.
