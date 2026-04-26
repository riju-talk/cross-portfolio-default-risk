# Loan Defaulter Prediction Study

This repository now contains a production-style ML workflow for default risk prediction using the **UCI Default of Credit Card Clients** dataset.

## What is implemented
- End-to-end training pipeline with leakage-safe preprocessing (`SimpleImputer` + `StandardScaler` in a `ColumnTransformer`).
- Benchmark comparison across three supervised learners:
  - Logistic Regression (class-weighted)
  - Random Forest (balanced subsampling)
  - XGBoost
- 5-fold stratified cross-validation for model selection (ROC-AUC).
- Holdout evaluation with threshold tuning based on best F1 score from precision-recall curve.
- Feature importance using permutation importance (ROC-AUC objective).
- Automated report generation:
  - `reports/model_report.md`
  - `reports/metrics.json`

## How to run
```bash
uv run python main.py
```

## Latest experimental results (April 26, 2026)
Cross-validation ROC-AUC (mean ± std):
- Logistic Regression: **0.7265 ± 0.0108**
- Random Forest: **0.7807 ± 0.0046**
- XGBoost: **0.7818 ± 0.0069**

Selected model on CV: **XGBoost**
- Holdout ROC-AUC: **0.7779**
- Holdout PR-AUC: **0.5521**
- F1: **0.5452**
- Precision: **0.5221**
- Recall: **0.5705**
- Tuned threshold: **0.2772**

## Research implication
The strongest signal comes from recent repayment-status trajectory variables (`pay_status_*`), not static demographics. This supports a research direction toward **temporal credit-behavior modeling** (e.g., sequence learning or time-to-default/survival frameworks) for earlier and more reliable delinquency intervention.
