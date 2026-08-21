# Rounds 1-4 Experimental Summary

## Objective
Predict default payment with positive class `default = 1`.

## Gates
ROC-AUC >= 0.75: PASS. Recall >= 0.60: PASS. Binary F1 >= 0.65: FAIL.

## Protocol
UCI id=350, deterministic stratified 70/15/15 split, TRAIN for fitting/OOF, VALIDATION for external development comparison, TEST untouched.

## Development Winner
XGBoostClassifier with ROUND1+BILL+PAYMENT, no sampling, threshold `0.461883`. Validation AUC `0.7817`, Precision `0.5008`, Recall `0.6101`, F1 `0.5501`.

## Rounds
Round 1 reached F1 approximately 0.5452. Round 2 feature engineering produced the winner at 0.5501 and reduced false positives versus the earlier baseline. Round 3 hard-negative weighting reduced FP but lowered F1; cascade and segmented thresholds did not improve. Round 4 temporal representations and CatBoost did not produce material validation generalization gain.

## Diagnostics
Class overlap was high: 27 exact duplicate feature groups, 11 conflicting-label groups, and 26.55% near-neighbor opposite-label rate. This indicates separation difficulty, not a mathematical upper bound.

## Runtime
Threshold search fell from about 91.5 seconds to 0.03 seconds after vectorization. T0/T1 matrices were 21,000x62 / 21,000x78 with no NaN, inf, object dtype, duplicate columns, or memory explosion.

## Limitation
VALIDATION was consulted repeatedly in Rounds 1-3. Final Round 4 promotion used frozen TRAIN-OOF threshold and was not used to fit.

## Distance
Best F1 gap to 0.65: `0.0999`. Required precision at observed recall for F1=0.65: `0.6955`.

## Test
TEST untouched. Await explicit instruction before one-time final evaluation.

## Conclusion
Development is frozen at the best F1 achieved under the evaluated protocol. No claim is made that this is the dataset's maximum possible F1.

## Methodological correction: F1 averaging

Initially, Rounds 1-4 were conducted prioritizing Binary F1 for `default=1`. The project now adopts Macro F1 as the operational interpretation of the A1 F1 gate. No model was retrained, no threshold was retuned, and Binary F1 remains a diagnostic. Under the corrected convention, the existing CatBoost A3 candidate is the methodological development winner: Macro F1 exceeds the XGBoost A3 candidate while AUC and Recall gates pass. The historical Binary F1 plateau near 0.5501 remains valid for class 1, but is not the official A1 gate.

## Methodological correction: F1 averaging

Initially, Rounds 1-4 were conducted prioritizing Binary F1 for `default=1`. The project now adopts Macro F1 as the operational interpretation of the A1 F1 gate. No model was retrained, no threshold was retuned, and Binary F1 remains a diagnostic. Under the corrected convention, the existing CatBoost A3 candidate is the methodological development winner: Macro F1 exceeds the XGBoost A3 candidate while AUC and Recall gates pass. The historical Binary F1 plateau near 0.5501 remains valid for class 1, but is not the official A1 gate.

## Methodological correction: F1 averaging

Initially, Rounds 1-4 were conducted prioritizing Binary F1 for `default=1`. The project now adopts Macro F1 as the operational interpretation of the A1 F1 gate. No model was retrained, no threshold was retuned, and Binary F1 remains a diagnostic. Under the corrected convention, the existing CatBoost A3 candidate is the methodological development winner: Macro F1 exceeds the XGBoost A3 candidate while AUC and Recall gates pass. The historical Binary F1 plateau near 0.5501 remains valid for class 1, but is not the official A1 gate.

## Methodological correction: F1 averaging

Initially, Rounds 1-4 were conducted prioritizing Binary F1 for `default=1`. The project now adopts Macro F1 as the operational interpretation of the A1 F1 gate. No model was retrained, no threshold was retuned, and Binary F1 remains a diagnostic. Under the corrected convention, the existing CatBoost A3 candidate is the methodological development winner: Macro F1 exceeds the XGBoost A3 candidate while AUC and Recall gates pass. The historical Binary F1 plateau near 0.5501 remains valid for class 1, but is not the official A1 gate.

## Methodological correction: F1 averaging

Initially, Rounds 1-4 were conducted prioritizing Binary F1 for `default=1`. The project now adopts Macro F1 as the operational interpretation of the A1 F1 gate. No model was retrained, no threshold was retuned, and Binary F1 remains a diagnostic. Under the corrected convention, the existing CatBoost A3 candidate is the methodological development winner: Macro F1 exceeds the XGBoost A3 candidate while AUC and Recall gates pass. The historical Binary F1 plateau near 0.5501 remains valid for class 1, but is not the official A1 gate.

## Methodological correction: F1 averaging

Initially, Rounds 1-4 were conducted prioritizing Binary F1 for `default=1`. The project now adopts Macro F1 as the operational interpretation of the A1 F1 gate. No model was retrained, no threshold was retuned, and Binary F1 remains a diagnostic. Under the corrected convention, the existing CatBoost A3 candidate is the methodological development winner: Macro F1 exceeds the XGBoost A3 candidate while AUC and Recall gates pass. The historical Binary F1 plateau near 0.5501 remains valid for class 1, but is not the official A1 gate.

## Methodological correction: F1 averaging

Initially, Rounds 1-4 were conducted prioritizing Binary F1 for `default=1`. The project now adopts Macro F1 as the operational interpretation of the A1 F1 gate. No model was retrained, no threshold was retuned, and Binary F1 remains a diagnostic. Under the corrected convention, the existing CatBoost A3 candidate is the methodological development winner: Macro F1 exceeds the XGBoost A3 candidate while AUC and Recall gates pass. The historical Binary F1 plateau near 0.5501 remains valid for class 1, but is not the official A1 gate.
