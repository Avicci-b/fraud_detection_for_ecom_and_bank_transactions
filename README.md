# Fraud Detection for E-Commerce and Bank Transactions

A machine learning project for **Adey Innovations Inc.** focused on detecting fraudulent e-commerce purchases and bank credit card transactions.

The project explores transaction behavior, feature engineering, imbalanced classification, model evaluation, and explainable machine learning to support better fraud detection decisions.

## Business Problem

Fraud detection involves balancing two important risks:

* **False positives:** legitimate transactions incorrectly flagged as fraudulent, creating customer friction.
* **False negatives:** fraudulent transactions incorrectly classified as legitimate, potentially causing financial losses.

The goal is to evaluate machine learning approaches that distinguish fraudulent from legitimate transactions while considering class imbalance, model performance, and business impact.

## Datasets

The project uses three datasets:

* `Fraud_Data.csv` — e-commerce transactions with fraud labels.
* `IpAddress_to_Country.csv` — IP address ranges mapped to countries for geolocation analysis.
* `creditcard.csv` — bank credit card transactions with fraud labels.

**Data availability:** Dataset files are not tracked in Git. Place the required files in the appropriate local data directories before running the analysis.

## Initial Project Structure

```text
fraud_detection_for_ecom_and_bank_transactions/
├── .github/
│   └── workflows/
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── scripts/
├── src/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

*The tree above describes the intended organization; adjust individual filenames and folders to match the current repository.*

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Avicci-b/fraud_detection_for_ecom_and_bank_transactions.git
cd fraud_detection_for_ecom_and_bank_transactions
```

### 2. Create and activate a virtual environment

On Linux or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the datasets

Obtain the required datasets from their authorized sources and place them in the locations expected by the notebooks. Check the notebook paths before running the analysis.

### 5. Run the analysis

Open the project in Jupyter Notebook or VS Code and run the notebooks in the appropriate order, beginning with exploratory data analysis and preprocessing.

## Key Findings — Task 1

### E-commerce Transactions

The dataset has an observed fraud rate of approximately **9.36%**.

* **Device sharing:** `device_shared_count` showed a strong relationship with fraud in the analysis. The observed fraud rate increased substantially for devices associated with many user accounts.
* **Time since signup:** Fraudulent purchases had a median time since signup of approximately one second, compared with approximately 60 days for legitimate purchases.
* **Geolocation:** Some countries showed elevated observed fraud rates relative to the overall baseline. These patterns require contextual investigation and should not be interpreted as evidence that a country's users are inherently fraudulent.
* **Purchase value and age:** These features showed no meaningful difference between fraud and legitimate transactions in the initial analysis.

### Bank Credit Card Transactions

The dataset has an observed fraud prevalence of approximately **0.17%**, making class imbalance a major modeling challenge.

* **Transaction amount:** The amount distribution was strongly right-skewed; a log transformation was explored to reduce skewness.
* **Fraud amount patterns:** Fraudulent transactions showed a different distribution of transaction amounts from legitimate transactions, including a lower median but a higher mean in the initial analysis.
* **PCA-transformed features:** Features V1–V28 are transformed variables, so their individual meanings are not directly interpretable from their names alone.
* **Resampling considerations:** Oversampling can change the class distribution used during training. Synthetic examples must be distinguished from real observations, and evaluation should be performed on data that retains the original distribution.

## Project Progress

* [x] **Task 1 — Data Analysis and Preprocessing:** Completed
* [ ] **Task 2 — Model Building and Training:** In progress
* [ ] **Task 3 — Model Explainability:** Not started

The modeling stage focuses on comparing classification approaches using suitable evaluation metrics for imbalanced data. The explainability stage will investigate important features and individual model predictions.

## Limitations and Future Work

* Evaluate models using metrics suitable for imbalanced classification, including precision, recall, F1-score, and precision-recall AUC.
* Compare baseline and ensemble models and document hyperparameter tuning.
* Examine false positives and false negatives in the context of business costs.
* Use feature importance and SHAP to investigate model behavior.
* Assess the limitations of synthetic oversampling and avoid evaluating on resampled test data.

## Author

**Biniyam Mitiku**
Information Science Student | Addis Ababa University

GitHub: [@Avicci-b](https://github.com/Avicci-b)
