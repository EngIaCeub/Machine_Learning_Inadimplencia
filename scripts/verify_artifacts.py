"""Mechanical final-delivery artifact checker.

Use only at S06/final delivery. Missing final artifacts are expected during architecture stage.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FINAL = [
    ROOT / "notebooks" / "credit_default_colab.ipynb",
    ROOT / "artifacts" / "notebook_executed.pdf",
    ROOT / "artifacts" / "report.pdf",
    ROOT / "artifacts" / "final_submission.pdf",
]

for path in EXPECTED_FINAL:
    state = "PASS" if path.exists() and path.stat().st_size > 0 else "MISSING"
    print(f"{state}: {path.relative_to(ROOT)}")
