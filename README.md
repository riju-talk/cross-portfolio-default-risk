<div align="center">

# 🏦 Cross-Portfolio Default Risk Prediction

### A Dual-Study Data Science Investigation: Credit Cards & Installment Loans

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-✓-green?style=flat)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-✓-orange?style=flat)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-✓-blue?style=flat)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Research%20Use-red?style=flat)]()

This repository presents a **research-oriented, data-science-first** investigation into credit default prediction across two heterogeneous risk domains. Every modeling decision flows from exploratory analysis, domain-driven feature engineering, rigorous diagnostics, and business-aware evaluation — not just benchmark chasing.

</div>

---

## 🚀 Key Results at a Glance

<div align="center">

| Dataset | Best Model (ROC-AUC) | Best Model (PR-AUC) | Max Business Savings |
|:-------:|:--------------------:|:-------------------:|:--------------------:|
| **Credit Card** (UCI, 30,000) | 🥇 **XGBoost — 0.7769** | 🥇 **XGBoost — 0.5421** | 💰 **$18.74M / 18,641% ROI** (Random Forest) |
| **LendingClub Loan** (5,000) | 🥇 **Random Forest — 0.7467** | 🥇 **Random Forest — 0.4011** | 📈 Not quantified (methodology transfers) |

</div>

- **Card default rate:** 22.13% &nbsp;•&nbsp; **Loan default rate:** 18.38% &nbsp;•&nbsp; **Total models evaluated:** 13 (8 Card + 5 Loan) &nbsp;•&nbsp; **Features engineered:** 31+ (27 Card + 4+ Loan)

---

## 📖 Overview

Two heterogeneous credit risk domains, one transferable analytical framework:

| Dimension | 💳 Credit Card (UCI) | 💰 Installment Loans (LendingClub) |
|---|---|---|
| **Population** | Taiwanese bank clients | US-based LendingClub borrowers |
| **Sample size** | 30,000 (23,972 train / 5,993 test) | 5,000 (subset; full dataset 2M+) |
| **Default rate** | 22.13% (imbalance 3.52:1) | 18.38% (imbalance 4.0:1) |
| **Features** | 23 original → 50 after engineering | 25 selected (150+ original, after leakage removal) |
| **Feature type** | Behavioral (payment history, bills) | Origination (credit profile, loan terms) |
| **Key challenge** | Multicollinearity in bill amounts | Leakage from post-origination fields |
| **Paper benchmark** | Yeh & Lien (2009) — replicated | Industry AUC benchmarks |

Both share a common **~20% default rate** but differ fundamentally in feature structure — making this a true test of methodological transferability.

---

## 📂 Repository Structure

<pre>
├── notebooks/
│   ├── 01_card_default_EDA.ipynb                        # Card: EDA with VIF, correlations, leakage audit
│   ├── 02_card_preprocessing_feature_engineering.ipynb  # Card: 27 engineered features + sorting smoothing
│   ├── 03_card_modeling.ipynb                           # Card: 8-model zoo + isotonic calibration
│   ├── 04_card_evaluation_analysis.ipynb                # Card: SHAP, cost curves, statistical tests
│   ├── 01_loan_default_EDA.ipynb                        # Loan: chunked EDA, missingness, target mapping
│   ├── 02_loan_preprocessing_feature_engineering.ipynb  # Loan: leakage-safe pipeline
│   ├── 03_loan_modeling.ipynb                           # Loan: 5 models + hyperparameter tuning
│   └── 04_loan_evaluation_analysis.ipynb                # Loan: threshold analysis, calibration, SHAP
├── pipelines/
│   ├── card_pipeline.py                                 # Card replication pipeline (3 models)
│   ├── loan_pipeline.py                                 # Loan replication pipeline (3 models)
│   └── diagnostics.py                                   # Shared visualization utilities
├── main.py                                              # Unified CLI entrypoint
├── data/
│   ├── raw/                                             # Raw source datasets (Git LFS tracked)
│   ├── usable/                                          # Preprocessed usable copies
│   └── processed/                                       # Notebook-generated artifacts
├── results/                                             # All metrics, plots, predictions
│   ├── card/                                            # Card track outputs (evaluation_output/, plots/)
│   └── loan/                                            # Loan track outputs
├── artifacts/                                           # Pipeline-generated metrics JSON
├── papers/                                              # Reference academic papers
└── CREDIT_RISK_ANALYSIS_REPORT.md                       # Full 538-line results interpretation
</pre>

---

## 🔬 Data Science Methodology

### Phase 1: Exploratory Data Analysis (`01_*_EDA.ipynb`)

**Card track** — 12 analysis sections including:
- Structural validation (shape, missingness, duplicates, data types)
- Feature taxonomy (demographic, repayment, bill, payment groups)
- Target analysis (22.13% default rate — why accuracy is misleading)
- **Leakage audit** (all features verified pre-target)
- Bivariate analysis with t-tests (defaulters vs non-defaulters)
- Correlation matrix + intra-group analysis & **multicollinearity quantification (VIF)**
- Payment behavior deep dive (zero-payment rates, payment-to-bill ratios)

**Loan track** — Chunked loading for memory efficiency:
- Terminal status target mapping (Charged Off / Default → 1, Fully Paid → 0)
- Imbalance quantification (18.38% default rate)
- Missingness analysis (concentrated in credit history features)
- **Leakage column identification — 5 groups, 145+ columns removed** (repayment history, dates, delinquency flags, hardship, settlement)

### Phase 2: Feature Engineering & Preprocessing (`02_*_preprocessing*.ipynb`)

**Card track** — Domain-driven engineering creates **27 new signals**:
- **Utilization ratios** (BILL_AMT / LIMIT_BAL per month) — captures credit stress
- **Payment ratios** (PAY_AMT / BILL_AMT per month) — repayment effort
- **Trend features** (linear slope of bills/payments over 6 months)
- **Delinquency aggregates** (NUM_DELAYED, MAX_DELAY, RECENCY_DELAY)
- **Log transforms** for skewed credit limits + binning (AGE_GROUP, UTIL_DECILE)
- **Sorting Smoothing Method** (Yeh & Lien 2009 replication) — real PD per client (bin size 500, 60 bins, PD range 7.4%–73.55%)

**Loan track** — Leakage-aware preprocessing:
- 145+ post-outcome columns removed (verified against `columns_to_drop.txt`)
- Percentage string parsing (int_rate, revol_util)
- Tiered missing value strategy (drop >50%, median-impute numeric, mode-impute categorical)
- Domain features: loan-to-income, payment-to-income, FICO average, delinquency flags
- Stratified train-test split (4,000 / 1,000) | StandardScaler for linear models

### Phase 3: Modeling (`03_*_modeling.ipynb`)

**Card track** — **8 models** with hyperparameter tuning + isotonic calibration:
Logistic Regression, LDA, Gaussian Naive Bayes, k-NN, Decision Tree, Random Forest, XGBoost, MLP
- RandomizedSearchCV (3-fold, **PR-AUC scoring**)
- Isotonic calibration on held-out validation split (Brier score comparison)
- Model comparison radar, ROC/PR overlays, probability distribution histograms

**Loan track** — 5 models with class-weight balancing:
Logistic Regression (baseline), Random Forest, XGBoost, LightGBM, SVM (dataset permitting)
- Hyperparameter finetuning with PR-AUC optimization

### Phase 4: Evaluation & Interpretation (`04_*_evaluation*.ipynb`)

**Card track** — 10 analysis sections:
- Paper benchmark replication (Yeh & Lien 2009)
- Statistical model comparison (Friedman test, critical difference diagram)
- **Cost-sensitive threshold optimization** (FN:FP ratios 10:1 → 500:1)
- SHAP explainability (beeswarm + waterfall plots)
- Error analysis (confusion matrices, model disagreement matrix)
- Calibration deep-dive (reliability diagrams)
- Ensemble strategy (simple/weighted averaging, **stacking**)
- **Business metrics** (expected value, savings, ROI)

**Loan track** — ROC/PR curves, threshold impact analysis, feature importance, calibration assessment (reliability diagram), prediction distribution analysis.

---

## 📊 Key Findings

### 💳 Credit Card Default (UCI Dataset) — 8 models, isotonic-calibrated

<div align="center">

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **🥇 XGBoost** | **0.7769** | **0.5421** | 0.4477 | **0.6603** | 0.3386 | **0.1367** |
| 🥈 Random Forest | 0.7758 | 0.5323 | **0.4547** | 0.6293 | **0.3560** | 0.1383 |
| MLP Neural Net | 0.7664 | 0.5339 | 0.4444 | 0.6549 | 0.3363 | 0.1387 |
| Decision Tree | 0.7670 | 0.5259 | 0.4435 | 0.6478 | 0.3371 | 0.1387 |
| KNN | 0.7556 | 0.5050 | 0.4298 | 0.6232 | 0.3281 | 0.1420 |
| LDA | 0.7528 | 0.4900 | 0.4227 | 0.6094 | 0.3235 | 0.1432 |
| Logistic Regression | 0.7526 | 0.4879 | 0.4052 | 0.6078 | 0.3039 | 0.1430 |
| Naive Bayes | 0.7099 | 0.4547 | 0.4088 | 0.6120 | 0.3069 | 0.1475 |

</div>

**Key insights:**
1. **XGBoost dominates ranking & calibration** — best ROC-AUC (0.7769), best PR-AUC (0.5421), best Brier (0.1367)
2. **Random Forest maximizes business value** — highest calibrated F1 (0.4547) and **$18,741,000 (18,641% ROI)** in estimated annual savings
3. **Top predictors** (SHAP consensus): `PAY_0` (most recent delinquency), `MAX_DELAY`, `NUM_DELAYED`, credit utilization ratios (`BILL_AMT / LIMIT_BAL`)

### 🏆 Ensemble Results (Card)

| Ensemble | ROC-AUC | Verdict |
|---|:---:|---|
| Simple Average | 0.7741 | Good baseline |
| Weighted Average | 0.7742 | Marginal improvement |
| **Stacking (LR meta)** | **0.7760** | **Best balance of performance & calibration** |

### 💰 LendingClub Loan Default — 5,000 loans, 25 features, 18.38% default rate

<div align="center">

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **🥇 Random Forest** | **0.7467** | **0.4011** | **0.4551** | 0.3695 | 0.5924 | 73.9% |
| 🥈 Logistic Regression | 0.7426 | 0.3949 | 0.4518 | 0.3333 | **0.7011** | 68.7% |
| XGBoost | 0.6923 | 0.3634 | 0.3842 | 0.3514 | 0.4239 | **75.0%** |

</div>

**Key insights:**
1. **Random Forest wins on small data** — built-in regularization (bagging + feature subsampling) beats boosting at 5K rows
2. **⚠️ XGBoost overfits severely**: train ROC-AUC **0.9957** vs validation **0.6923** (gap = 0.3034) — needs more data or stronger regularization (`max_depth=3, reg_lambda=5.0, min_child_weight=10, n_estimators=200`)
3. **Grade alone stratifies risk 10.7×**: A = 5.4% default → G = 58.04%
4. **Top predictors**: `int_rate` (0.307 correlation), `loan_to_income`, `fico_range_low` — LendingClub's internal risk pricing is the strongest signal

### 🔗 Cross-Study Insights

1. **Recent payment behavior dominates both domains** — PAY_0 (card) ≈ interest rate (loan)
2. **Utilization matters everywhere** — credit utilization for cards, debt-to-income for loans
3. **Calibration is critical** — raw probabilities systematically over-estimate risk; isotonic correction is essential (Naive Bayes Brier: 0.324 → 0.147)
4. **PR-AUC over ROC-AUC** — with ~20% default rates, PR-AUC better reflects real-world detection ability
5. **Ensembles stabilize** — stacking (0.7760) is production-recommended, though the gain over XGBoost alone (0.7769) shows diminishing returns

---

## 💼 Business Impact Assessment

Cost-sensitive threshold optimization (FN:FP 10:1 → 500:1) shows the **default 0.5 threshold is suboptimal** for every model — cost-optimal thresholds range from **0.15–0.35**.

| Model | Savings (USD) | ROI (%) |
|:---|:---:|:---:|
| **Random Forest** | **$18,741,000** | **18,641%** |
| XGBoost | ~$17.5M (est.) | ~17,500% |
| Baseline (no model) | $0 | — |

*Basis: 5,993 test clients extrapolated to portfolio scale; ~$10K average loss given default.*

---

## ⚙️ Reproducibility Protocol

### Environment

```bash
uv sync        # Python 3.12+ required, deps via pyproject.toml
```

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

## 🧠 Why Data Science Over ML Engineering

- **Hypothesis-first investigation** — EDA drives every preprocessing and modeling decision
- **Domain-appropriate features** — credit utilization ratios, delinquency trends, sorting-smoothing PD estimates
- **Multi-faceted evaluation** — not just AUC: calibration, cost-sensitivity, SHAP, business metrics
- **Cross-dataset methodology** — transferable analytical frameworks, not just one tuned model
- **Paper replication** — Yeh & Lien (2009) sorting smoothing reproduced and extended
- **Leakage awareness** — rigorous post-outcome feature removal (145+ columns in loan data)
- **Visualization as evidence** — every plot answers a specific business/research question

---

## 📥 Data Sources

1. **UCI Default of Credit Card Clients** — Yeh & Lien (2009). 30,000 clients, 23 features. `python data/dataset_download_ucimlrepo.py`
2. **LendingClub Accepted Loans** — 2007–2018Q4, ~2M+ records, 150+ features. `python data/dataset_download_kaggle.py`

Large files tracked via Git LFS. Review upstream licensing before redistribution.

---

## 📚 References

- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
- LendingClub statistics. (2018). Accepted and rejected loan data, 2007–2018Q4.
- UCI Machine Learning Repository. Default of Credit Card Clients Dataset.

---

<div align="center">
<sub>Report generated June 2026 · Analysis period: Sept 2005 (Cards), 2007–2018 (Loans) · Full details in <a href="CREDIT_RISK_ANALYSIS_REPORT.md">CREDIT_RISK_ANALYSIS_REPORT.md</a></sub>
</div>
