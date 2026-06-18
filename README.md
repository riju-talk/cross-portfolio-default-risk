# Credit Risk Default Prediction: A Dual-Study Data Science Investigation

This repository presents a **research-oriented, data-science-first** investigation into credit default prediction across two heterogeneous risk domains:

1. **Credit Card Defaults** (UCI Taiwan dataset) — 30,000 clients, 23 behavioral features
2. **Installment Loan Defaults** (LendingClub) — 2M+ loans, 150+ origination features

The philosophy is **data science over ML engineering**: every modeling decision flows from exploratory analysis, domain-driven feature engineering, rigorous diagnostics, and business-aware evaluation — not just benchmark chasing.

---

## Research Design

### Cross-Dataset Validation Mindset

| Dimension | Credit Card (UCI) | Installment Loans (LendingClub) |
|---|---|---|
| **Population** | Taiwanese bank clients | US-based LendingClub borrowers |
| **Sample size** | 30,000 (fixed) | ~2M+ (chunked) |
| **Default rate** | ~22% | ~20% |
| **Feature type** | Behavioral (payment history, bills) | Origination (credit profile, loan terms) |
| **Temporal aspect** | 6-month lookback | Varies by loan origination |
| **Key challenge** | Multicollinearity in bill amounts | Leakage from post-origination fields |
| **Paper benchmark** | Yeh & Lien (2009) | Industry AUC benchmarks |

Both share a common **~20% default rate** but differ fundamentally in feature structure — making this a true test of methodological transferability.

---

## Repository Structure

```
├── notebooks/
│   ├── 01_card_default_EDA.ipynb               # Card: comprehensive EDA with VIF, correlations
│   ├── 02_card_preprocessing_feature_engineering.ipynb  # Card: 12 engineered features + sorting smoothing
│   ├── 03_card_modeling.ipynb                  # Card: 8-model zoo + isotonic calibration
│   ├── 04_card_evaluation_analysis.ipynb       # Card: SHAP, cost curves, statistical tests
│   ├── 01_loan_default_EDA.ipynb        # Loan: chunked EDA, missingness, target mapping
│   ├── 02_loan_preprocessing_feature_engineering.ipynb  # Loan: leakage-safe pipeline
│   ├── 03_loan_modeling.ipynb           # Loan: 5 models + hyperparameter tuning
│   └── 04_loan_evaluation_analysis.ipynb # Loan: threshold analysis, calibration, SHAP
├── pipelines/
│   ├── card_pipeline.py                 # Card replication pipeline (3 models)
│   ├── loan_pipeline.py                 # Loan replication pipeline (3 models)
│   └── diagnostics.py                   # Shared visualization utilities
├── main.py                              # Unified CLI entrypoint
├── data/
│   ├── raw/                             # Raw source datasets (Git LFS tracked)
│   ├── usable/                          # Preprocessed usable copies
│   └── processed/                       # Notebook-generated artifacts
├── results/                             # All metrics, plots, predictions
│   ├── card/                            # Card track outputs
│   └── loan/                            # Loan track outputs
└── papers/                              # Reference academic papers
```

---

## Data Science Methodology

### Phase 1: Exploratory Data Analysis (`01_*_EDA.ipynb`)

**Card track** — 12 analysis sections including:
- Structural validation (shape, missingness, duplicates, data types)
- Feature taxonomy (demographic, repayment, bill, payment groups)
- Target analysis (22% default rate, why accuracy is misleading)
- Leakage audit (all features verified pre-target)
- Univariate distributions by feature group
- Bivariate analysis with t-tests (defaulters vs non-defaulters)
- Full correlation matrix + intra-group analysis
- Multicollinearity quantification (Variance Inflation Factor)
- Payment behavior deep dive (zero-payment rates, payment-to-bill ratios)

**Loan track** — Chunked loading for memory efficiency, covering:
- Terminal status target mapping (Charged Off / Default vs Fully Paid)
- Imbalance quantification (~20% default rate)
- Missingness analysis (concentrated in credit history features)
- Feature distribution analysis by default status
- Leakage column identification (5 groups: repayment, dates, delinquency, hardship, settlement)

### Phase 2: Feature Engineering & Preprocessing (`02_*_preprocessing*.ipynb`)

**Card track** — Domain-driven feature engineering creates 12 new signals:
- **Utilization ratios** (BILL_AMT / LIMIT_BAL per month)
- **Payment ratios** (PAY_AMT / BILL_AMT per month)
- **Trend features** (linear slope of bills and payments over 6 months)
- **Delinquency features** (count delayed, max delay, recency)
- **Log transforms** for skewed credit limits
- **Sorting Smoothing Method** (Yeh & Lien 2009 replication) estimating real PD per client
- 8 publication-quality visualizations including risk segmentation matrix and mutual information analysis

**Loan track** — Leakage-aware preprocessing:
- 114+ post-outcome columns removed (verified against `columns_to_drop.txt`)
- Percentage string parsing (int_rate, revol_util)
- Tiered missing value strategy (drop >50%, median-impute numeric, mode-impute categorical)
- Domain features: loan-to-income, payment-to-income, FICO average, delinquency flags
- Stratified train-test split | StandardScaler for linear models

### Phase 3: Modeling (`03_*_modeling.ipynb`)

**Card track** — 8 models with hyperparameter tuning + isotonic calibration:
- Logistic Regression, LDA, Gaussian Naive Bayes, k-NN
- Decision Tree, Random Forest, XGBoost, MLP Neural Network
- RandomizedSearchCV (3-fold, PR-AUC scoring)
- Isotonic calibration on held-out validation split
- Calibration impact analysis (Brier score comparison)
- Model comparison radar, ROC/PR overlays, probability distribution histograms

**Loan track** — 5 models with class-weight balancing:
- Logistic Regression (baseline, interpretable)
- Random Forest (non-linear relationships, feature importance)
- XGBoost (state-of-the-art, scale_pos_weight for imbalance)
- LightGBM (efficient, when available)
- SVM (RBF kernel, when dataset permits)
- Hyperparameter finetuning with PR-AUC optimization

### Phase 4: Evaluation & Interpretation (`04_*_evaluation*.ipynb`)

**Card track** — Deep evaluation with 10 analysis sections:
- Paper benchmark replication (Yeh & Lien 2009)
- Statistical model comparison (Friedman test, critical difference diagram)
- Cost-sensitive threshold optimization (FN:FP ratios from 10:1 to 500:1)
- SHAP explainability (beeswarm + waterfall plots)
- Error analysis (confusion matrices, model disagreement matrix)
- Calibration deep-dive (reliability diagrams)
- Ensemble strategy (simple/weighted averaging, stacking)
- Business metrics (expected value, savings, ROI)

**Loan track** — Practical evaluation including:
- ROC and Precision-Recall curves per model
- Threshold impact analysis (precision/recall/F1 vs threshold)
- Confusion matrix and classification report
- Feature importance analysis (top predictors of default)
- Calibration assessment (reliability diagram)
- Prediction distribution analysis

---

## Reproducibility Protocol

### Environment

```bash
uv sync
```

Python 3.12+ required. Dependencies managed via `pyproject.toml`.

### Run Pipelines

```bash
# Both problems
python main.py --problem both

# Credit card only
python main.py --problem card

# Loan only with sample control
python main.py --problem loan --max-loan-rows 120000 --loan-chunksize 100000
```

### Suggested Notebook Order

1. `01_card_default_EDA.ipynb` / `01_loan_default_EDA.ipynb`
2. `02_card_preprocessing_feature_engineering.ipynb` / `02_loan_preprocessing_feature_engineering.ipynb`
3. `03_card_modeling.ipynb` / `03_loan_modeling.ipynb`
4. `04_card_evaluation_analysis.ipynb` / `04_loan_evaluation_analysis.ipynb`

Then run script pipelines for clean replication:
```bash
python main.py --problem both --max-loan-rows 30000 --loan-chunksize 50000
```

---

## Key Findings

### Credit Card Default (UCI Dataset)

| Model | ROC-AUC | PR-AUC | F1 | Top-Decile Capture |
|---|---|---|---|---|
| XGBoost | 0.7761 | 0.5557 | 0.4666 | 0.3090 |
| Random Forest | 0.7652 | 0.5410 | 0.4452 | 0.3012 |
| Logistic Regression | 0.6973 | 0.4175 | 0.4158 | 0.2418 |

**Top predictors:** PAY_0 (most recent delinquency), AVG_UTIL (credit utilization), RECENCY_DELAY

### LendingClub Loan Default

| Model | ROC-AUC | PR-AUC | F1 | Top-Decile Capture |
|---|---|---|---|---|
| Random Forest | 0.7366 | 0.4207 | 0.4489 | 0.2549 |
| XGBoost | 0.7321 | 0.4153 | 0.3957 | 0.2611 |
| Logistic Regression | 0.6854 | 0.3521 | 0.4012 | 0.2321 |

**Top predictors:** interest rate, FICO score, debt-to-income ratio, loan-to-income ratio

### Cross-Study Insights

1. **Recent payment behavior dominates**: PAY_0 (card) ~ interest rate (loan) — the most recent signal is always strongest
2. **Utilization matters in both domains**: Credit utilization for cards, debt-to-income for loans
3. **Calibration is critical**: Raw probabilities systematically over-estimate risk — isotonic correction is essential
4. **PR-AUC over ROC-AUC**: With ~20% default rate, PR-AUC better reflects real-world detection ability
5. **Ensembles stabilize**: Simple averaging of top-3 models outperforms any single model on both datasets

---

## Why Data Science Over ML Engineering

This repository demonstrates **data science thinking** through:

- **Hypothesis-first investigation**: EDA drives every preprocessing and modeling decision
- **Domain-appropriate features**: Credit utilization ratios, delinquency trends, sorting smoothing PD estimates
- **Multi-faceted evaluation**: Not just AUC, but calibration, cost-sensitivity, SHAP, business metrics
- **Cross-dataset methodology**: Demonstrating transferable analytical frameworks, not just tuning one model
- **Paper replication**: Yeh & Lien (2009) sorting smoothing method reproduced and extended
- **Visualization as evidence**: Every plot answers a specific business/research question
- **Leakage awareness**: Rigorous post-outcome feature removal, especially critical in loan data

---

## Data Sources

1. **UCI Default of Credit Card Clients** — Yeh & Lien (2009). 30,000 clients, 23 features. Download: `python data/dataset_download_ucimlrepo.py`
2. **LendingClub Accepted Loans** — 2007-2018Q4, ~2M+ records, 150+ features. Download: `python data/dataset_download_kaggle.py`

Large files tracked via Git LFS. Review upstream licensing before redistribution.

---

## References

- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
- LendingClub statistics. (2018). Accepted and rejected loan data, 2007-2018Q4.
