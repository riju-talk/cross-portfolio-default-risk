## Datasets

This project uses two publicly available datasets for default prediction analysis:

### 1. LendingClub Loan Data

**Dataset Overview:**

This dataset contains loan-level data from the LendingClub peer-to-peer lending platform, one of the largest peer-to-peer lending marketplaces in the United States. The data includes comprehensive information about borrowers, loan characteristics, and loan outcomes.

**Dataset Characteristics:**
- **Source:** LendingClub via Kaggle
- **Subject Area:** Financial Services, Peer-to-Peer Lending
- **Associated Tasks:** Credit Risk Assessment, Loan Default Prediction
- **Instances:** Large-scale loan data spanning 2007-2018
- **Use Case:** Commonly used for credit risk modeling and default prediction research

**Content:**
The dataset contains borrower attributes (credit history, employment, income), loan characteristics (amount, term, interest rate, grade), and loan outcomes (current status, payment history, defaults). This comprehensive data enables detailed analysis of factors contributing to loan defaults in the peer-to-peer lending market.

**Note:** Due to licensing and size constraints, raw data files are not included in this repository. Use the `dataset_download.py` script to download the data.

---

### 2. Default of Credit Card Clients

**Dataset Overview:**

This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods.

**Dataset Characteristics:**
- **Type:** Multivariate
- **Subject Area:** Business, Credit Risk
- **Associated Tasks:** Classification, Default Prediction
- **Feature Type:** Integer, Real
- **Instances:** 30,000
- **Features:** 23
- **Missing Values:** No
- **Source:** UCI Machine Learning Repository (Donated on 1/25/2016)

**Dataset Information:**

This research focused on customers' default payments in Taiwan and compared the predictive accuracy of probability of default among six data mining methods. From the perspective of risk management, the result of predictive accuracy of the estimated probability of default is more valuable than the binary result of classification (credible or not credible clients). 

Because the real probability of default is unknown, this study presented the novel Sorting Smoothing Method to estimate the real probability of default. With the real probability of default as the response variable (Y), and the predictive probability of default as the independent variable (X), the simple linear regression result (Y = A + BX) shows that the forecasting model produced by artificial neural network has the highest coefficient of determination; its regression intercept (A) is close to zero, and regression coefficient (B) to one. Therefore, among the six data mining techniques, artificial neural network is the only one that can accurately estimate the real probability of default.

**Features:**

| Variable Name | Role | Type | Demographic | Description | Units | Missing Values |
|---------------|------|------|-------------|-------------|-------|----------------|
| ID | ID | Integer | | Customer ID | | no |
| X1 | Feature | Integer | | LIMIT_BAL (Credit Limit) | | no |
| X2 | Feature | Integer | Sex | SEX (Gender) | | no |
| X3 | Feature | Integer | Education Level | EDUCATION | | no |
| X4 | Feature | Integer | Marital Status | MARRIAGE | | no |
| X5 | Feature | Integer | Age | AGE | | no |
| X6 | Feature | Integer | | PAY_0 (Repayment Status Sept) | | no |
| X7 | Feature | Integer | | PAY_2 (Repayment Status Aug) | | no |
| X8 | Feature | Integer | | PAY_3 (Repayment Status July) | | no |
| X9 | Feature | Integer | | PAY_4 (Repayment Status June) | | no |
| X10 | Feature | Integer | | PAY_5 (Repayment Status May) | | no |
| X11 | Feature | Integer | | PAY_6 (Repayment Status April) | | no |
| X12 | Feature | Integer | | BILL_AMT1 (Bill Statement Sept) | | no |
| X13 | Feature | Integer | | BILL_AMT2 (Bill Statement Aug) | | no |
| X14 | Feature | Integer | | BILL_AMT3 (Bill Statement July) | | no |
| X15 | Feature | Integer | | BILL_AMT4 (Bill Statement June) | | no |
| X16 | Feature | Integer | | BILL_AMT5 (Bill Statement May) | | no |
| X17 | Feature | Integer | | BILL_AMT6 (Bill Statement April) | | no |
| X18 | Feature | Integer | | PAY_AMT1 (Payment Amount Sept) | | no |
| X19 | Feature | Integer | | PAY_AMT2 (Payment Amount Aug) | | no |
| X20 | Feature | Integer | | PAY_AMT3 (Payment Amount July) | | no |
| X21 | Feature | Integer | | PAY_AMT4 (Payment Amount June) | | no |
| X22 | Feature | Integer | | PAY_AMT5 (Payment Amount May) | | no |
| X23 | Feature | Integer | | PAY_AMT6 (Payment Amount April) | | no |
| Y | Target | Binary | | Default payment next month | | no |
