"""Verify that the approved architecture scaffold is present."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "AGENTS.md",
    ".codex/config.toml",
    ".codex/agents/architect.toml",
    ".codex/agents/ml-reviewer.toml",
    ".codex/agents/implementer.toml",
    ".codex/agents/verifier.toml",
    ".codex/agents/artifact-ops.toml",
    ".agents/skills/plan-spec/SKILL.md",
    ".agents/skills/data-pipeline/SKILL.md",
    ".agents/skills/model-experiment/SKILL.md",
    ".agents/skills/explain-monitor/SKILL.md",
    ".agents/skills/colab-repro/SKILL.md",
    ".agents/skills/artifact-qa/SKILL.md",
    "specs/INDEX.md",
    "specs/S00-project-contract.md",
    "specs/S01-architecture.md",
    "specs/S02-data-pipeline.md",
    "specs/S03-modeling.md",
    "specs/S04-explainability-demo.md",
    "specs/S05-monitoring.md",
    "specs/S06-delivery.md",
    "src/credit_default/config.py",
    "notebooks/credit_default_colab.ipynb",
]

missing = [rel for rel in REQUIRED if not (ROOT / rel).exists()]
if missing:
    raise SystemExit("Missing architecture files:\n- " + "\n- ".join(missing))

print(f"PASS: architecture scaffold contains {len(REQUIRED)} required files.")
