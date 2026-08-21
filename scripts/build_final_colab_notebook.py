"""Build the final Colab notebook from the approved project API."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "FINAL_COLAB_REPRODUCIBLE.ipynb"


def _normalize_cell_source(text: str) -> list[str]:
    lines = [line.removeprefix("        ") for line in text.splitlines()]
    return "\n".join(lines).strip().splitlines(True)


def markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _normalize_cell_source(text)}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _normalize_cell_source(text),
    }


CELLS = [
    markdown(
        """# Credit Default — Final Colab Reproducible\n\n
        REPRODUCTION_ONLY. This notebook reproduces the frozen A1 candidate.\n\n
        No tuning, model selection, threshold search, or feature experimentation is performed.
    """
    ),
    markdown(
        """## 1. Objective and A1 requirements\n\n
        The target is `default = 1`. The frozen candidate is CatBoostClassifier with A3 = ROUND1 + BILL + PAYMENT and threshold 0.247743.\n\n
        Official A1 gates: ROC-AUC >= 0.75, Recall(default=1) >= 0.60, and Macro F1 >= 0.65. Binary F1(default=1) remains a diagnostic.
    """
    ),
    markdown(
        """## 2. Environment setup\n\n
        Run this section once in a fresh CPU Colab runtime. The repository URL is the only publication-time field that may need filling.
    """
    ),
    code(
        """import os\n
        import subprocess\n
        from pathlib import Path\n\n
        REPO_URL = \"TODO_REPOSITORY_URL\"\n
        REPO_DIR = Path(\"ml-credit-default-architecture\")\n\n
        if not REPO_DIR.exists():\n
            if REPO_URL == \"TODO_REPOSITORY_URL\":\n
                raise RuntimeError(\"Fill REPO_URL once with the public read-only repository URL.\")\n
            subprocess.run([\"git\", \"clone\", REPO_URL, str(REPO_DIR)], check=True)\n\n
        %cd {REPO_DIR}\n
        %pip install -q -r requirements.txt catboost==1.2.10\n
        os.environ[\"PYTHONPATH\"] = str(Path.cwd() / \"src\")\n
        print(\"Environment setup complete.\")
    """
    ),
    markdown("## 3. Imports and reproducibility\n\nAll randomness is centralized at seed 42."),
    code(
        """import json\n
        import random\n
        import sys\n
        from pathlib import Path\n\n
        import catboost\n
        import matplotlib.pyplot as plt\n
        import numpy as np\n
        import pandas as pd\n
        import sklearn\n
        from catboost import CatBoostClassifier\n
        from sklearn.metrics import (\n
            ConfusionMatrixDisplay, average_precision_score, confusion_matrix,\n
            precision_recall_curve, roc_auc_score, roc_curve,\n
        )\n\n
        SEED = 42\n
        random.seed(SEED)\n
        np.random.seed(SEED)\n
        print(\"Python:\", sys.version)\n
        print(\"pandas:\", pd.__version__)\n
        print(\"numpy:\", np.__version__)\n
        print(\"scikit-learn:\", sklearn.__version__)\n
        print(\"catboost:\", catboost.__version__)\n
        print(\"Random seed:\", SEED)
    """
    ),
    markdown("## 4. Dataset loading\n\nThe official UCI loader downloads dataset id 350 programmatically. No manual upload or personal storage is used."),
    code(
        """sys.path.insert(0, str(Path.cwd() / \"src\"))\n
        from credit_default.config import get_project_config\n
        from credit_default.data.load import load_uci_dataset\n
        from credit_default.data.schema import validate_basic_shape\n
        from credit_default.data.split import validate_distinct_splits\n
        from credit_default.features.credit_default_features import (\n
            RAW_CATEGORICAL_COLUMNS, build_behavioral_features,\n
        )\n
        from credit_default.modeling.evaluate import evaluate_binary_classifier\n\n
        config = get_project_config()\n
        features, target = load_uci_dataset(dataset_id=350)\n
        validate_basic_shape(n_rows=len(features), n_features=features.shape[1])\n
        print(\"Dataset shape:\", features.shape)\n
        display(features.head())\n
        display(target.value_counts().sort_index().rename(\"count\").to_frame())\n
        display(features.isna().sum().sort_values(ascending=False).head(10).rename(\"missing\").to_frame())
    """
    ),
    markdown("## 5. Dataset overview and target definition\n\nThere are 23 original attributes. The target is normalized by the official loader and must contain only 0 and 1."),
    code(
        """assert len(features) >= 29000 and len(features) <= 31000\n
        assert set(target.unique()) == {0, 1}\n
        assert len(features) == len(target)\n
        assert not any(column in features.columns for column in (\"Y\", \"default\", \"target\"))\n
        print(\"Positive class: default = 1\")\n
        print(\"Original features:\", features.shape[1])
    """
    ),
    markdown("## 6. Frozen Train/Validation/Test split\n\nThe persisted Round 4 manifest is used directly. Test is the complement of the frozen Train and Validation indices; no new split is created."),
    code(
        """manifest_path = Path(\"artifacts/experiments/round4_split_manifest.json\")\n
        manifest = json.loads(manifest_path.read_text())\n
        train_index = pd.Index(manifest[\"train_indices\"], dtype=features.index.dtype)\n
        validation_index = pd.Index(manifest[\"validation_indices\"], dtype=features.index.dtype)\n
        test_index = features.index.difference(train_index.union(validation_index))\n\n
        assert train_index.intersection(validation_index).empty\n
        assert train_index.intersection(test_index).empty\n
        assert validation_index.intersection(test_index).empty\n
        X_train, y_train = features.loc[train_index], target.loc[train_index]\n
        X_validation, y_validation = features.loc[validation_index], target.loc[validation_index]\n
        X_test, y_test = features.loc[test_index], target.loc[test_index]\n
        split_table = pd.DataFrame({\n
            \"split\": [\"train\", \"validation\", \"test\"],\n
            \"rows\": [len(X_train), len(X_validation), len(X_test)],\n
            \"positive_rate\": [y_train.mean(), y_validation.mean(), y_test.mean()],\n
        })\n
        display(split_table)\n
        assert split_table[\"rows\"].tolist() == [21000, 4500, 4500]
    """
    ),
    markdown("## 7. Feature engineering A3\n\nA3 is exactly ROUND1 + BILL + PAYMENT. PAY remains ordinal/temporal; X2, X3, and X4 are categorical for CatBoost. No target is passed to feature engineering."),
    code(
        """A3_GROUPS = (\"round1\", \"bill\", \"payment\")\n
        X_train_a3 = build_behavioral_features(X_train, enabled_groups=A3_GROUPS)\n
        X_validation_a3 = build_behavioral_features(X_validation, enabled_groups=A3_GROUPS)\n
        X_test_a3 = build_behavioral_features(X_test, enabled_groups=A3_GROUPS)\n
        for frame in (X_train_a3, X_validation_a3, X_test_a3):\n
            assert not any(column in frame.columns for column in (\"Y\", \"default\", \"target\"))\n
            assert np.isfinite(frame.select_dtypes(include=\"number\").to_numpy()).all()\n
        print(\"Original features:\", features.shape[1])\n
        print(\"Final A3 features:\", X_train_a3.shape[1])\n
        print(\"Groups: ROUND1 + BILL + PAYMENT\")
    """
    ),
    markdown("## 8. Final model configuration\n\nConfiguration is loaded from the frozen winner artifact. No tuning or threshold search is performed."),
    code(
        """winner = json.loads(Path(\"artifacts/final/development_winner.json\").read_text())\n
        params = winner[\"hyperparameters\"].copy()\n
        FROZEN_THRESHOLD = float(winner[\"metrics\"][\"threshold\"])\n
        cat_columns = [column for column in RAW_CATEGORICAL_COLUMNS if column in X_train_a3.columns]\n
        for frame in (X_train_a3, X_validation_a3, X_test_a3):\n
            for column in cat_columns:\n
                frame[column] = frame[column].astype(str)\n
        model_summary = {\n
            \"model\": winner[\"model\"],\n
            \"feature_set\": winner[\"feature_set\"],\n
            \"hyperparameters\": params,\n
            \"seed\": SEED,\n
            \"categorical_features\": cat_columns,\n
            \"frozen_threshold\": FROZEN_THRESHOLD,\n
        }\n
        display(pd.Series(model_summary, dtype=object).to_frame(\"value\"))
    """
    ),
    markdown("## 9. Training\n\nThe frozen CatBoost candidate is fitted only on TRAIN. The model uses native categorical handling and CPU-safe settings."),
    code(
        """model = CatBoostClassifier(\n
            loss_function=\"Logloss\",\n
            eval_metric=\"AUC\",\n
            random_seed=SEED,\n
            verbose=False,\n
            thread_count=2,\n
            allow_writing_files=False,\n
            **params,\n
        )\n
        model.fit(X_train_a3, y_train, cat_features=cat_columns)\n
        print(\"Training complete.\")
    """
    ),
    markdown("## 10. Evaluation function\n\nThe same explicit positive-class metric semantics are used for Validation and Test. Scores are probabilities; predictions use only the frozen threshold."),
    code(
        """def evaluate_split(name, X_frame, y_frame):\n
            scores = model.predict_proba(X_frame)[:, 1]\n
            metrics = evaluate_binary_classifier(y_frame, scores, FROZEN_THRESHOLD)\n
            metrics[\"average_precision\"] = average_precision_score(y_frame, scores)\n
            metrics[\"split\"] = name\n
            return scores, metrics\n\n
        validation_scores, validation_metrics = evaluate_split(\"validation\", X_validation_a3, y_validation)\n
        test_scores, test_metrics = evaluate_split(\"test\", X_test_a3, y_test)\n
        metric_names = [\"roc_auc\", \"average_precision\", \"precision_positive\", \"recall_positive\", \"f1_binary_default_1\", \"f1_class_0\", \"f1_macro\", \"f1_weighted\", \"accuracy\", \"tn\", \"fp\", \"fn\", \"tp\"]\n
        display(pd.DataFrame([{\"metric\": key, \"validation\": validation_metrics[key], \"test\": test_metrics[key]} for key in metric_names]))
    """
    ),
    markdown("## 11. Validation evaluation\n\nValidation is shown before the final holdout to confirm the frozen candidate remains consistent with its reference evidence."),
    code(
        """validation_reference = {\"roc_auc\": 0.7811007964, \"recall_positive\": 0.6050251256, \"f1_macro\": 0.7024923058}\n
        display(pd.DataFrame([{\"metric\": key, \"reproduced\": validation_metrics[key], \"reference\": value, \"absolute_difference\": abs(validation_metrics[key] - value)} for key, value in validation_reference.items()]))
    """
    ),
    markdown("## 12. Final TEST evaluation\n\nTEST is evaluated once in this reproduction-only notebook with the frozen model and threshold. It is not used for any decision."),
    code(
        """test_display = {key: test_metrics[key] for key in metric_names}\n
        display(pd.Series(test_display, name=\"test\").to_frame())\n
        print(\"Frozen threshold =\", FROZEN_THRESHOLD)
    """
    ),
    markdown("## 13. A1 gate analysis\n\nThe official F1 gate is Macro F1. Binary F1(default=1) is retained as a diagnostic and is not hidden."),
    code(
        """gate_table = pd.DataFrame([\n
            {\"metric\": \"ROC-AUC\", \"value\": test_metrics[\"roc_auc\"], \"requirement\": \">= 0.75\", \"status\": test_metrics[\"roc_auc\"] >= 0.75},\n
            {\"metric\": \"Recall(default=1)\", \"value\": test_metrics[\"recall_positive\"], \"requirement\": \">= 0.60\", \"status\": test_metrics[\"recall_positive\"] >= 0.60},\n
            {\"metric\": \"Macro F1\", \"value\": test_metrics[\"f1_macro\"], \"requirement\": \">= 0.65\", \"status\": test_metrics[\"f1_macro\"] >= 0.65},\n
        ])\n
        gate_table[\"status\"] = gate_table[\"status\"].map({True: \"PASS\", False: \"FAIL\"})\n
        display(gate_table)\n
        print(\"Overall:\", (gate_table[\"status\"] == \"PASS\").sum(), \"/ 3 PASS\")\n
        assert (gate_table[\"status\"] == \"PASS\").all()
    """
    ),
    markdown("## 14. Confusion matrix\n\nThe matrix is generated from the reproduced TEST predictions at the frozen threshold."),
    code(
        """test_predictions = (test_scores >= FROZEN_THRESHOLD).astype(int)\n
        cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])\n
        ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(cmap=\"Blues\", values_format=\"d\")\n
        plt.title(\"TEST Confusion Matrix\")\n
        plt.show()\n
        print(\"TN, FP, FN, TP =\", tuple(cm.ravel()))
    """
    ),
    markdown("## 15. ROC curve\n\nROC-AUC uses probability scores and does not select a threshold."),
    code(
        """fpr, tpr, _ = roc_curve(y_test, test_scores)\n
        plt.plot(fpr, tpr, label=f\"AUC = {test_metrics['roc_auc']:.4f}\")\n
        plt.plot([0, 1], [0, 1], \"--\", color=\"gray\")\n
        plt.xlabel(\"False positive rate\")\n
        plt.ylabel(\"True positive rate\")\n
        plt.title(\"TEST ROC Curve\")\n
        plt.legend()\n
        plt.show()
    """
    ),
    markdown("## 16. Precision-Recall curve\n\nThe PR curve is diagnostic. The frozen threshold is marked; no new threshold is searched."),
    code(
        """precision, recall, thresholds = precision_recall_curve(y_test, test_scores)\n
        plt.plot(recall, precision, label=f\"AP = {test_metrics['average_precision']:.4f}\")\n
        frozen_point = test_metrics[\"recall_positive\"], test_metrics[\"precision_positive\"]\n
        plt.scatter(*frozen_point, color=\"red\", label=f\"frozen threshold = {FROZEN_THRESHOLD:.6f}\")\n
        plt.xlabel(\"Recall\")\n
        plt.ylabel(\"Precision\")\n
        plt.title(\"TEST Precision-Recall Curve\")\n
        plt.legend()\n
        plt.show()
    """
    ),
    markdown("## 17. Model explainability\n\nCatBoost global feature importance is computed from the fitted frozen model. Importance indicates association with model decisions, not causality."),
    code(
        """importance = pd.DataFrame({\n
            \"feature\": X_train_a3.columns,\n
            \"importance\": model.get_feature_importance(),\n
        }).sort_values(\"importance\", ascending=False).head(15)\n
        display(importance)\n
        importance.sort_values(\"importance\").plot.barh(x=\"feature\", y=\"importance\", legend=False, figsize=(8, 6))\n
        plt.title(\"Top 15 CatBoost Feature Importance\")\n
        plt.show()\n
        print(\"Groups represented: PAY history, BILL behavior, PAYMENT behavior, and A3 aggregates.\")
    """
    ),
    markdown(
        """## 18. Monitoring strategy\n\n
        Monitor schema, missing values, invalid categories, and expected ranges as data quality checks. Monitor feature distributions with PSI, KS, or an equivalent agreed measure; monitor the distribution of p(default=1) and predicted positive rate for prediction drift. When labels arrive, monitor ROC-AUC, Recall(default=1), Macro F1, Precision, and the confusion matrix. Persistent values below AUC 0.75, Recall 0.60, or Macro F1 0.65 should trigger investigation and revalidation; an isolated fluctuation does not automatically require retraining.
    """
    ),
    markdown("## 19. Pipeline visual\n\nDataset → Schema validation → Frozen split → Feature engineering A3 → CatBoostClassifier → predict_proba → Frozen threshold → A1 metrics → Final evaluation"),
    markdown("## 20. Experimental history summary\n\nThe complete development rounds are persisted in the project artifacts. This notebook reproduces only the final selected pipeline."),
    code(
        """history = pd.read_csv(\"artifacts/final/a1_macro_f1_reranking.csv\")\n
        history_columns = [column for column in [\"model\", \"feature_set\", \"f1_macro\", \"roc_auc\", \"recall\"] if column in history.columns]\n
        display(history.loc[:, history_columns].head(10))
    """
    ),
    markdown("## 21. Reproducibility check\n\nReference values are used only for a diagnostic comparison. Reproduced metrics are calculated above from the dataset, frozen split, frozen features, frozen model, and frozen threshold."),
    code(
        """reference = {\"roc_auc\": 0.7865276746, \"recall_positive\": 0.6104417671, \"f1_macro\": 0.7019172459, \"f1_binary_default_1\": 0.5502262443}\n
        repro_table = pd.DataFrame([{\"metric\": key, \"reproduced\": test_metrics[key], \"reference\": value, \"absolute_difference\": abs(test_metrics[key] - value), \"status\": \"PASS\" if abs(test_metrics[key] - value) <= 0.01 else \"REVIEW\"} for key, value in reference.items()])\n
        display(repro_table)
    """
    ),
    markdown(
        """## 22. Final conclusion\n\n
        The frozen CatBoostClassifier A3 candidate was reproduced without tuning or selection. Under A1, the reproduced TEST evaluation is expected to satisfy 3/3 gates. The holdout is not written to official artifacts by this notebook. The second physical TEST access from the historical evaluator was an infrastructure recovery and is documented in the project audit.
    """
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "colab": {"name": "FINAL_COLAB_REPRODUCIBLE.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(notebook, indent=2).replace(
        "TODO_REPOSITORY_URL",
        "https://github.com/EngIaCeub/Machine_Learning_Inadimplencia.git",
    )
    serialized = serialized.replace(
        'if REPO_URL == \\\"https://github.com/EngIaCeub/Machine_Learning_Inadimplencia.git\\\":',
        'if False:',
    )
    NOTEBOOK_PATH.write_text(serialized, encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()

