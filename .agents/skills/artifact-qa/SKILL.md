---
name: artifact-qa
description: Perform low-cost mechanical QA for required files, links, notebook outputs, exports, placeholders, naming, and final delivery package.
---

Read S06 only.

Verify mechanically:
- path exists
- file non-empty
- expected extension/name
- required links populated
- no TODO/PLACEHOLDER in final deliverables
- notebook has executed outputs when required
- report/notebook PDF/final PDF exist

Do not evaluate ML quality. Return PASS/FAIL with exact paths.
