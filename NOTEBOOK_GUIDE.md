# Loan Defaulter Prediction Study - Complete Notebook Documentation

## Project Structure

This project implements a comprehensive machine learning pipeline for loan default prediction using two datasets:
1. **LendingClub Loan Data** (2007-2018) - Large-scale peer-to-peer lending data
2. **UCI Credit Card Default** - 30,000 credit card clients from Taiwan

---

## Notebook Organization

The project contains **8 complete notebooks** organized into 4-step workflows for each dataset:

### Loan Default Prediction Workflow

#### 01_loan_default_EDA.ipynb
**Purpose**: Exploratory Data Analysis for LendingClub accepted loans
**Key Content**:
- Dataset overview and structure
- Target variable definition (default vs non-default)
- Class imbalance analysis with visualizations
- Leakage column identification and removal
- Feature type analysis and distributions
- Correlation heatmaps
- Missing value analysis

#### 02_loan_preprocessing_feature_engineering.ipynb
**Purpose**: Transform raw loan data into modeling-ready format
**Key Content**:
- Chunked data loader for large files (memory-efficient)
- Leakage-aware column removal
- Feature categorization (loan, borrower, credit history)
- Missing value imputation strategies
- Categorical encoding (one-hot, ordinal)
- Train-test splitting
- Feature scaling for sensitive algorithms

#### 03_loan_modeling.ipynb
**Purpose**: Train and compare multiple classification models
**Key Content**:
- **5 Classifiers**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM
- Class-weight balancing for imbalance
- Stratified train-test splits
- ROC-AUC and PR-AUC evaluation
- Feature importance analysis
- Model comparison table
- Best model persistence

#### 04_loan_evaluation_analysis.ipynb
**Purpose**: Comprehensive model evaluation and interpretation
**Key Content**:
- ROC and Precision-Recall curves
- Confusion matrices
- Threshold sensitivity analysis
- Feature importance visualization
- Prediction distribution analysis
- Calibration curves
- Business recommendations
- Limitations and future work

---

### Credit Card Default Prediction Workflow

#### 01_card_default_EDA.ipynb
**Purpose**: Exploratory Data Analysis for UCI credit card dataset
**Key Content**:
- Dataset characteristics (30K rows, 23 features, no missing values)
- Feature descriptions and interpretations
- Target distribution (default rate ~22%)
- Demographic, repayment, bill, and payment feature analysis
- Clean data validation

#### 02_card_preprocessing_feature_engineering.ipynb
**Purpose**: Preprocess and engineer features for credit card data
**Key Content**:
- Feature renaming (X1→LIMIT_BAL, etc.)
- Feature grouping (demographics, repayment status, bills, payments)
- Data quality checks
- **Feature engineering**: avg bill/payment amounts, utilization rate, payment ratios, delayed payment counts
- Stratified train-test split (80/20)
- Feature scaling with StandardScaler
- Save processed datasets

#### 03_card_modeling.ipynb
**Purpose**: Train 5 classifiers on credit card data
**Key Content**:
- **5 Classifiers**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network (MLP)
- Class balancing via weights
- Model comparison on ROC-AUC and PR-AUC
- Feature importance for tree-based models
- Best model selection and persistence

#### 04_card_evaluation_analysis.ipynb
**Purpose**: Detailed evaluation and cross-dataset insights
**Key Content**:
- ROC and PR curves
- Confusion matrix and classification reports
- Threshold analysis with precision-recall-F1 trade-offs
- Feature importance (repayment status features dominate)
- Prediction distribution histograms
- Calibration analysis
- **Cross-dataset validation discussion** (loan vs card, US vs Taiwan)
- Production deployment recommendations
- Research extension ideas

---

## Key Design Principles

### 1. Leakage Prevention
All notebooks strictly enforce that only **pre-origination** features are used:
- Remove repayment aggregates (total_pymnt, recoveries, etc.)
- Remove post-issuance dates (last_pymnt_d, settlement_date, etc.)
- Remove delinquency indicators observed after issuance
- Remove hardship and settlement information

### 2. Class Imbalance Handling
- Use `class_weight='balanced'` in sklearn models
- Use `scale_pos_weight` in XGBoost/LightGBM
- Evaluate with **ROC-AUC** and **PR-AUC**, not accuracy
- Visualize threshold trade-offs for business decision-making

### 3. Reproducibility
- Random seeds set to 42 throughout
- Preprocessing functions saved in notebooks for reuse
- Train-test splits preserved to disk
- Best models saved as `.pkl` files
- Results tables saved as CSV

### 4. Interpretability
- Feature importance plots for tree models
- Coefficient inspection for logistic regression
- SHAP analysis mentioned for future work
- Clear markdown narrative explaining each step

### 5. Scalability
- Chunked loading for large LendingClub files
- Memory-efficient processing (drop unused columns early)
- Parallel processing enabled (`n_jobs=-1`)
- Sample-based demonstrations with clear notes on full-scale processing

---

## Data Processing Summary

### LendingClub Loan Data
- **Raw**: ~2M+ rows, 150+ columns
- **After leakage removal**: ~114 columns
- **Terminal statuses only**: Charged Off, Default, Fully Paid
- **Target**: Binary (1=default, 0=non-default)
- **Default rate**: ~15-20% (imbalanced)

### UCI Credit Card Data
- **Raw**: 30,000 rows, 24 columns
- **No missing values**
- **Target**: Binary (1=default, 0=non-default)
- **Default rate**: ~22%
- **Engineered features**: +5 (avg bill/payment, ratios, delayed counts)

---

## Model Performance Expectations

Based on literature and dataset characteristics:

### Loan Default (Expected)
- **ROC-AUC**: 0.65-0.75 (good discrimination)
- **PR-AUC**: 0.30-0.50 (minority class performance)
- **Best models**: XGBoost, LightGBM (handle non-linearity)

### Credit Card Default (Expected)
- **ROC-AUC**: 0.75-0.80 (cleaner data, fewer features)
- **PR-AUC**: 0.40-0.60
- **Best models**: XGBoost, Random Forest

---

## Running the Notebooks

### Prerequisites
```bash
# Install dependencies (already in pyproject.toml)
pip install pandas numpy scikit-learn xgboost lightgbm plotly seaborn jupyter
```

### Execution Order

**For Loan Default:**
1. Run `01_loan_default_EDA.ipynb` (lightweight, uses existing processed data)
2. Run `02_loan_preprocessing_feature_engineering.ipynb` (creates processed sample)
3. Run `03_loan_modeling.ipynb` (trains 5 models, ~2-5 minutes)
4. Run `04_loan_evaluation_analysis.ipynb` (generates visualizations)

**For Credit Card Default:**
1. Run `01_card_default_EDA.ipynb` (fast, small dataset)
2. Run `02_card_preprocessing_feature_engineering.ipynb` (creates train/test splits)
3. Run `03_card_modeling.ipynb` (trains 5 models, ~1-2 minutes)
4. Run `04_card_evaluation_analysis.ipynb` (comprehensive evaluation)

**Notes:**
- LendingClub notebooks use a 10K-row sample by default for speed
- To process the full dataset, uncomment chunked loader calls in preprocessing notebook
- Credit card notebooks process all 30K rows (fast enough)

---

## Output Artifacts

### Saved Files
```
data/processed/
├── credit_card_processed.csv       # Full processed UCI data
├── credit_card_train.csv           # Training split
├── credit_card_test.csv            # Test split
└── loan_sample_processed.csv       # LendingClub sample

results/
├── loan_model_comparison.csv       # Model metrics comparison
├── card_model_comparison.csv       # Model metrics comparison
├── best_loan_model_xgb.pkl         # Saved best loan model
└── best_card_model_xgb.pkl         # Saved best card model
```

---

## Research Context

This project is **inspired by ACM research on credit risk modeling** and follows best practices:

1. **Leakage-aware preprocessing** (Kaufman et al., 2012)
2. **Cost-sensitive learning** for imbalanced data
3. **Ranking-based evaluation** (ROC-AUC, PR-AUC over accuracy)
4. **Feature interpretability** for regulatory compliance
5. **Cross-dataset validation** to assess generalization

### Differences from Production Systems
- Simplified feature engineering (no external bureau data)
- No hyperparameter optimization (conservative defaults)
- No temporal cross-validation
- No fairness audits
- No online learning / model updating

---

## Limitations and Extensions

### Current Limitations
1. Sample-based processing for LendingClub (demo purposes)
2. Simplified imputation (median/mode only)
3. No SHAP/LIME individual explanations
4. No cost-matrix optimization
5. No time-based cross-validation

### Future Extensions
1. **Advanced feature engineering**: Payment velocity, trend features, polynomial interactions
2. **Hyperparameter tuning**: GridSearchCV, Bayesian optimization
3. **Model stacking/ensembling**: Combine predictions from multiple models
4. **SHAP analysis**: Explain individual predictions for interpretability
5. **Fairness analysis**: Check for demographic bias
6. **Transfer learning**: Apply loan model to card data and measure performance drop
7. **Temporal validation**: Time-series cross-validation with walk-forward splits
8. **Production pipeline**: MLOps setup with monitoring, A/B testing, retraining

---

## Citation and Acknowledgments

### Datasets
- **LendingClub**: Available via Kaggle (historical peer-to-peer lending data)
- **UCI Credit Card**: Yeh, I-Cheng. (2016). Default of Credit Card Clients. UCI Machine Learning Repository.

### References
- Kaufman, S., Rosset, S., & Perlich, C. (2012). Leakage in data mining. ACM TKDD.
- Provost, F., & Fawcett, T. (2001). Robust classification for imprecise environments. Machine Learning.

---

## Contact and Support

For questions or contributions:
- Review individual notebook markdown cells for detailed explanations
- Check inline code comments for implementation details
- Refer to scikit-learn and XGBoost documentation for model parameters
- Consult ACM KDD proceedings for credit risk modeling research

---

**Project Status**: ✅ Complete
**Last Updated**: January 2026
**Python Version**: 3.12+
**Key Dependencies**: scikit-learn 1.8+, xgboost 3.1+, pandas 3.0+, plotly 6.5+
