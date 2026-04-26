# Loan Defaulter Prediction Study

This project builds a **credit default risk classifier** on the UCI `default_of_credit_card_clients` dataset and compares multiple supervised learning algorithms in a leakage-aware pipeline.

## Problem framing

Default prediction is an imbalanced binary classification task where false negatives are costly (missed high-risk borrowers), while false positives reduce loan approval rate and customer experience. This repository focuses on:

- reproducible data ingestion and experiment execution,
- robust preprocessing with imputation + scaling/encoding,
- model comparison with ranking and classification metrics,
- practical risk metrics (e.g., top-decile capture rate).

## Dataset

- **Source file**: `data/default_of_credit_card_clients.csv`
- **Rows**: 30,000
- **Features used**: 23 (excluding target and ID)
- **Default rate**: 22.12%

## Implemented ML pipeline

`main.py` performs the following:

1. Loads and validates the dataset and target column.
2. Removes non-predictive ID.
3. Stratified train/test split (80/20, random seed 42).
4. Preprocessing with `ColumnTransformer`:
   - numeric: median imputation + standard scaling,
   - categorical: mode imputation + one-hot encoding.
5. Trains and evaluates:
   - Logistic Regression (class-weighted),
   - Random Forest (class-balanced subsampling),
   - XGBoost (if available).
6. Writes experiment artifact to `artifacts/metrics.json`.

## Current benchmark results

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Top-decile capture |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.7758 | **0.5514** | **0.5432** | 0.5055 | 0.5870 | **0.3097** |
| XGBoost | **0.7774** | 0.5511 | 0.4655 | **0.6553** | 0.3610 | 0.3082 |
| Logistic Regression | 0.7081 | 0.4904 | 0.4613 | 0.3672 | **0.6202** | 0.2969 |

> Best model by PR-AUC (with ROC-AUC tie-break): **Random Forest**.

## Research implication (brainstorm)

A key research implication is that **ranking performance and operational capture metrics may diverge from threshold metrics**:

- XGBoost achieved the highest ROC-AUC, but Random Forest produced better PR-AUC/F1 and top-decile capture.
- For credit operations, this suggests that model selection should be aligned to **portfolio triage objectives** (e.g., “capture as many true defaulters in top-risk bucket”) rather than only global discrimination.
- A publishable follow-up is to evaluate **decision-curve analysis / expected profit curves** under varying default costs, approval rates, and interest margins, instead of metric-only comparison.

## Run

```bash
.venv/bin/python main.py
```

The run creates:

- `artifacts/metrics.json` with dataset stats, model metrics, and classification reports.

## Next ML-engineering steps

- Add `src/` package layout + unit tests for data validation and metric functions.
- Track experiments with MLflow/W&B and hyperparameter search (Optuna).
- Add calibration and reject-inference workflow for deployment realism.
