---
name: colab-repro
description: Verify that the Colab/notebook can execute from a fresh environment and reproduce required data, metrics, plots, explanations, and demo outputs.
---

Read S06.

Check:
- install/bootstrap cells
- automatic UCI load
- no absolute paths or credentials
- deterministic config
- thin notebook calling reusable source functions
- full top-to-bottom execution
- output/metric regeneration
- exact package lock captured only after a green fresh run

Prefer automated checks. Return concise failures with cell/file location.
