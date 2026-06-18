# Loan Defaulter Prediction Study — Complete Notebook Documentation

## Project Structure

A **data-science-first** investigation into credit default across two domains:
1. **LendingClub Installment Loans** (2007-2018, 2M+ records, 150+ features)
2. **UCI Credit Card Default** (30,000 Taiwanese clients, 23 behavioral features)

---

## Notebook Organization

8 notebooks organized as 4-step research workflows per dataset.

---

### Loan Default Prediction Workflow

#### `01_loan_default_EDA.ipynb` (20 cells)
**Purpose**: Exploratory Data Analysis for LendingClub accepted loans
- Terminal-status target mapping (Charged Off / Default → 1, Fully Paid → 0)
- Class imbalance quantification (~20% default rate)
- Leakage audit: 5 groups identified (repayment aggregates, dates, delinquency, hardship, settlement) — 145+ columns
- Univariate distribution analysis (loan_amnt, int_rate, annual_inc, dti, revol_util, fico_avg × target overlay)
- Bivariate analysis with t-tests (int_rate, dti, annual_inc by default status)
- Correlation heatmap (full Spearman matrix, top target correlations)
- Missingness analysis (concentrated in mort_acc, pub_rec_bankruptcies)
- Payment capacity analysis (loan_to_income, payment_to_income, fico_avg)
- EDA artifact export to `results/loan/eda_summary.json`

#### `02_loan_preprocessing_feature_engineering.ipynb` (20 cells)
**Purpose**: Transform raw loan data into leakage-safe, modeling-ready format
- Chunked loader for memory-efficient large-file processing
- 5 leakage groups removed (114+ columns from `columns_to_drop.txt`)
- Feature categorization: loan, borrower, credit history, loan attributes
- Domain-driven feature engineering (12 features):
  - `loan_to_income`, `payment_to_income`, `fico_avg`, `fico_score_bucket`
  - `acc_open_ratio`, `inquiry_per_acc`, `has_delinq`, `has_pub_rec`, `has_bankruptcy`
  - `dti_bin`, `int_rate_bin`, `emp_length_numeric`
- Tiered missing value strategy: drop >50%, median-impute numeric, mode-impute categorical
- Ordinal encoding (grade A→1…G→7, sub_grade A1→1…G5→35)
- One-hot encoding (home_ownership, purpose, verification_status, initial_list_status, application_type)
- Outlier capping at 99th percentile (annual_inc, loan_amnt, revol_bal, installment)
- Mutual information analysis (top-20 features plotted)
- Stratified 80/20 split, StandardScaler, artifact export

#### `03_loan_modeling.ipynb` (22 cells)
**Purpose**: Train, tune, calibrate, and compare 4 classifiers
- **4 models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- Hyperparameter tuning: `RandomizedSearchCV` with 3-fold stratified CV (PR-AUC scoring)
- Tuned parameters: LR (C, solver), RF (n_estimators, max_depth, min_samples_leaf/split, max_features), XGB (n_estimators, max_depth, lr, subsample, colsample, min_child_weight, reg_lambda), LGBM (n_estimators, max_depth, lr, num_leaves, min_child_samples)
- Probability calibration: isotonic regression on 80/20 holdout split
- Raw vs calibrated comparison: ROC-AUC, PR-AUC, Brier score
- Model comparison table sorted by calibrated PR-AUC
- ROC curves overlay with AUC legend
- PR curves overlay with prevalence baseline
- Calibration impact bar chart (Brier Δ)
- Feature importance comparison (1×3 subplots for tree models)
- Artifact export: model_comparison.csv, all_predictions.npz, best_params.json, models/*.pkl, training_summary.json

#### `04_loan_evaluation_analysis.ipynb` (24 cells)
**Purpose**: Comprehensive evaluation with SHAP, cost analysis, and business metrics
- ROC curve overlay (all models, AUC values)
- PR curve overlay (all models, PR-AUC values)
- Confusion matrix heatmaps (sensitivity, specificity per model)
- Cost-sensitive threshold optimization (FN:FP = 100:1, find optimal threshold per model)
- Calibration deep-dive: reliability diagrams per model with Brier annotation
- SHAP explainability (TreeSHAP): beeswarm plot (top 15 features), waterfall plots (correct default prediction + false negative analysis)
- Feature importance rank agreement heatmap across tree models
- Prediction distribution analysis (default vs non-default overlay, KS statistic)
- Business metrics: expected loss without/with model, cost savings, ROI
- Summary artifact export to `results/loan_evaluation_summary.json`

---

### Credit Card Default Prediction Workflow

#### `01_card_default_EDA.ipynb` (24 cells)
**Purpose**: Exploratory Data Analysis for UCI credit card clients
- Structural validation: (30,000 × 24), zero missing, zero duplicates, all-numeric
- Feature taxonomy: demographic, repayment, bill, payment groups
- Target distribution: ~22% default rate (why accuracy is misleading)
- Leakage audit: all features verified pre-target (April–Sept 2005, target = Oct 2005)
- Univariate histograms by feature group
- Bivariate analysis with t-tests (LIMIT_BAL by default status)
- Full correlation matrix + intra-group analysis (bill amounts: high multicollinearity)
- Variance Inflation Factor (VIF) quantification
- Payment behavior deep dive (payment-to-bill ratio, zero-payment rates)
- EDA artifact export to `results/card/eda_summary.json`

#### `02_card_preprocessing_feature_engineering.ipynb` (28 cells)
**Purpose**: Transform raw data with 12 engineered features + Sorting Smoothing PD estimation
- 12 engineered features:
  - Utilization ratios (UTIL_1…6 = BILL_AMT / LIMIT_BAL)
  - Payment ratios (PAY_RATIO_1…6 = PAY_AMT / BILL_AMT, capped at 3)
  - Averages (AVG_BILL_AMT, AVG_PAY_AMT)
  - Trend slopes (BILL_TREND, PAY_TREND via row-wise linear regression)
  - Delinquency features (NUM_DELAYED, MAX_DELAY, RECENCY_DELAY)
  - Summaries (AVG_UTIL, AVG_PAY_RATIO) + LIMIT_BAL_LOG
- **8 publication-quality visualizations**:
  1. Credit utilization risk curve (default rate by decile, 95% CI)
  2. Delinquency heatmap (non-default vs default temporal patterns)
  3. Payment discipline trajectory (monthly payment ratios with Q25–Q75 bands)
  4. Risk segmentation matrix (LIMIT_BAL × AVG_UTIL heatmap)
  5. Mutual information (model-agnostic signal ranking)
  6. Sorting Smoothing Method (Yeh & Lien 2009 replication — real PD estimation)
  7. Real PD distribution & calibration gap
  8. Feature engineering impact (cumulative MI comparison)
- Stratified 80/20 split, StandardScaler, artifact export

#### `03_card_modeling.ipynb` (26 cells)
**Purpose**: 8-model zoo with hyperparameter tuning and isotonic calibration
- **8 models**: Logistic Regression, LDA, Gaussian Naive Bayes, k-NN, Decision Tree, Random Forest, XGBoost, MLP Neural Network
- `RandomizedSearchCV` (3-fold, PR-AUC scoring) for tunable models
- Isotonic calibration on held-out validation split
- Raw vs calibrated metrics: ROC-AUC, PR-AUC, F1, Precision, Recall
- Model comparison table (sorted by calibrated ROC-AUC)
- 6 visualizations:
  1. Model performance radar (raw vs calibrated ROC/PR-AUC)
  2. Calibration impact (Brier score comparison)
  3. ROC curves overlay (all 8 models)
  4. Precision-Recall curves overlay
  5. Predicted probability distributions (2×4 subplots)
  6. Feature importance comparison (3 tree models side-by-side)
- Artifact export: model_comparison.csv, all_predictions.npz, best_params.json, training_summary.json

#### `04_card_evaluation_analysis.ipynb` (31 cells)
**Purpose**: Deep evaluation with paper replication, SHAP, ensembles, and business impact
- Paper benchmark replication (Yeh & Lien 2009 — ANN, LR, RF, XGB literature baselines)
- Statistical model comparison: Friedman test, Nemenyi post-hoc, critical difference diagram
- Cost-sensitive threshold optimization (4 scenarios: FN:FP from 10:1 to 500:1)
- Cost curves visualization (total cost vs threshold per scenario)
- SHAP explainability: beeswarm plots (tree models), feature importance rank agreement heatmap
- Error analysis: confusion matrices (all 8 models), model disagreement matrix
- Calibration deep-dive: reliability diagrams (2×4 grid, raw vs calibrated Brier)
- Ensemble strategy: simple average, weighted average, stacking (logistic regression meta-learner)
- Business metrics: expected value, cost savings, ROI at optimal threshold
- Final artifact export for presentation

---

## Key Design Principles

1. **Leakage Prevention**: All notebooks remove post-outcome features (repayment aggregates, dates, delinquency, hardship, settlement fields in loan data; target-month information in card data)
2. **Imbalance Handling**: `class_weight='balanced'`, `scale_pos_weight`, isotonic calibration, PR-AUC over accuracy
3. **Reproducibility**: seed=42, stratified splits, artifact persistence, deterministic pipelines
4. **Interpretability**: SHAP, feature importance across models, calibration analysis, cost-sensitive thresholds
5. **Scalability**: chunked loading for large files, memory-efficient processing, `n_jobs=-1`

---

## Data Summary

| | LendingClub Loans | UCI Credit Card |
|---|---|---|
| **Rows** | ~2M+ (sample: 180K) | 30,000 |
| **Features** | 150+ raw → ~30 after leakage removal + engineering | 23 → 35 after 12 engineered |
| **Default rate** | ~20% | ~22% |
| **Key predictors** | int_rate, fico_avg, dti, loan_to_income | PAY_0, AVG_UTIL, MAX_DELAY |
| **Challenge** | Leakage, missingness, scale | Multicollinearity, limited features |

---

## Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm plotly seaborn shap statsmodels jupyter joblib
```

Or via uv:
```bash
uv sync
```

---

## Quick Start

```bash
# Run both pipeline scripts
python main.py --problem both --max-loan-rows 30000

# Or run notebooks in order per track (see above)
```

---

## Output Artifacts

```
results/
├── card/
│   ├── metrics.json
│   ├── plots/ (dataset/ + 7 plots per model)
│   └── predictions/ (CSV per model)
├── loan/
│   ├── metrics.json
│   ├── plots/ (dataset/ + 7 plots per model)
│   └── predictions/ (CSV per model)
├── card_model_comparison.csv
├── loan_model_comparison.csv
└── best_*_model_*.pkl
```

---

## Research References

- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for predictive accuracy of probability of default. *Expert Systems with Applications*.
- Kaufman, S., Rosset, S., & Perlich, C. (2012). Leakage in data mining. *ACM TKDD*.
- Provost, F., & Fawcett, T. (2001). Robust classification for imprecise environments. *Machine Learning*.

---

**Python**: 3.12+ | **Key deps**: scikit-learn 1.8+, xgboost 3.1+, pandas 3.0+, plotly 6.5+
