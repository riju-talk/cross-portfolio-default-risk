# Loan defaulter prediction using data driven methods

Loan default prediction is a fundamental problem in credit risk assessment, where early identification of high-risk borrowers can support informed lending and risk management decisions. With the increasing availability of large-scale financial data, machine learning methods have become widely adopted for modeling default risk due to their ability to capture complex, non-linear relationships.

This project implements a supervised machine learning pipeline for loan default prediction using public loan-level data, inspired by prior ACM research on credit risk modeling. The study focuses on leakage-aware preprocessing, feature engineering across borrower and loan attributes, and handling class imbalance through cost-sensitive learning.

To evaluate robustness and generalization, experiments are conducted across two heterogeneous financial datasets: peer-to-peer installment loan data and an academic credit card default benchmark. Model performance is assessed using ranking-based metrics, and interpretability is emphasized through analysis of key drivers influencing default risk
