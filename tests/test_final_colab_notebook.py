import ast
import json
from pathlib import Path

NOTEBOOK = Path("notebooks/FINAL_COLAB_REPRODUCIBLE.ipynb")


def test_final_colab_notebook_is_valid_and_linear():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 20

    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    full_text = "\n".join(sources)
    required_sections = (
        "Environment setup",
        "Dataset loading",
        "Frozen Train/Validation/Test split",
        "Feature engineering A3",
        "Final model configuration",
        "Validation evaluation",
        "Final TEST evaluation",
        "A1 gate analysis",
        "Confusion matrix",
        "ROC curve",
        "Precision-Recall curve",
        "Model explainability",
        "Monitoring strategy",
        "Reproducibility check",
    )
    assert all(section in full_text for section in required_sections)
    assert "https://github.com/EngIaCeub/Machine_Learning_Inadimplencia.git" in full_text
    assert "TODO_REPOSITORY_URL" not in full_text
    assert not any(
        value in full_text
        for value in ("C:\\Users\\", "C:/Users/", "OneDrive", "Área de Trabalho", "API_KEY", "PASSWORD")
    )

    code_cells = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
        ast.parse(source)
        code_cells += 1
    assert code_cells >= 18


def test_notebook_references_frozen_only_behavior():
    full_text = "".join("".join(cell.get("source", [])) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
    assert "CatBoostClassifier" in full_text
    assert "A3_GROUPS = (\"round1\", \"bill\", \"payment\")" in full_text
    assert "FROZEN_THRESHOLD" in full_text
    assert "RandomizedSearchCV" not in full_text
    assert "GridSearchCV" not in full_text
    assert "final_test_predictions.csv" not in full_text
