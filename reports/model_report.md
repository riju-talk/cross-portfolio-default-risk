# Loan Default Prediction Report

## Data & Task
- Dataset: UCI credit card default dataset (`30,000` rows, `23` predictors, binary target).
- Objective: classify whether a customer defaults the following month.

## Cross-Validation ROC-AUC (5-fold)
- **logistic_regression**: 0.7265 ± 0.0108
- **random_forest**: 0.7807 ± 0.0046
- **xgboost**: 0.7818 ± 0.0069

## Selected Model: xgboost
- ROC-AUC: **0.7779**
- PR-AUC: **0.5521**
- F1 (threshold-tuned): **0.5452**
- Precision: **0.5221**
- Recall: **0.5705**
- Decision threshold: **0.2772**
- Confusion matrix (TN, FP / FN, TP): **[3980, 693] / [570, 757]**

## Top Predictive Drivers (Permutation Importance on ROC-AUC)
- pay_status_sep: 0.07705
- credit_limit: 0.01964
- bill_amt_sep: 0.01831
- pay_amt_aug: 0.00792
- pay_amt_jul: 0.00651
- pay_amt_sep: 0.00603
- pay_status_aug: 0.00440
- pay_status_jul: 0.00362

## Research Implication
Repayment status trajectories (`pay_status_*`) dominate feature importance, suggesting that dynamic payment behavior is more informative than static demographics. This implies future work should model temporal delinquency patterns explicitly (for example with sequence models or survival analysis) to improve early-warning performance.