# Project State

phase: final_academic_delivery_ready

approved:
- agent architecture
- model routing policy
- caveman mode
- context economy strategy
- clean architecture light
- specs/skills split

status:
- S02 = DONE
- S03 = ROUND4_FINAL_PLATEAU
- A1 = VALIDATION_GATES_REACHED
- Final TEST = A1_FINAL_TEST_GATES_REACHED
- Rounds 1-4 development freeze = DONE
- Final academic delivery = READY
- Final Colab notebook = READY; public repository URL pending publication
- Colab link provided: https://colab.research.google.com/drive/1_Jgx1HXY3wPmAU8GJB8B347WKp0kumSV

test:
- CLOSED
- do not execute until gate resolution

best_validation_candidate:
- model: CatBoostClassifier
- feature_set: A3
- roc_auc: 0.7811
- binary_f1_default_1: 0.5498
- macro_f1: 0.7025
- weighted_f1: 0.7877
- recall_default_1: 0.6050
- precision_default_1: 0.5038
- threshold: 0.247743
- confusion_matrix: TN=2912 FP=593 FN=393 TP=602

pending_decision:
- none; A1 official F1 convention = Macro F1

notes:
- A1 correction applied: official gate F1 = Macro F1; Binary F1 remains diagnostic
- historical Binary F1 winner remains XGBoost A3 at 0.5501
- methodological winner after correction: CatBoost A3 at Macro F1 0.7025
- no models retrained and no thresholds retuned during A1 correction
- development winner persisted at artifacts/final/development_winner.json
- final experimental summary persisted at artifacts/final/rounds_1_to_4_summary.md
- gates: ROC-AUC PASS, Recall PASS, Macro F1 PASS (3/3)
- best F1 under evaluated protocol: 0.5501; this is not a mathematical upper bound
- explicit human instruction required before one-time final TEST evaluation
- final TEST evaluated with frozen CatBoost A3; AUC/Recall/Macro F1 gates PASS
- final TEST metrics persisted under artifacts/final/final_test_metrics.json
- deterministic evaluator recovery occurred after first prediction call; no model or threshold change
- final academic documents persisted under docs/ and delivery/
- final Colab notebook and README persisted under notebooks/
- post-test tuning: NO
- final dependency lock must be generated after first green Colab run
- notebook must remain thin
- UCI id=350 live smoke verified on 2026-08-18
- Gate 2 EDA coverage verified on 2026-08-18
- S03 amendment approved on 2026-08-18: HistGradientBoostingClassifier allowed as extra validation-only candidate
- S03-v3 amendment approved on 2026-08-18: XGBoostClassifier allowed as extra validation-only candidate
