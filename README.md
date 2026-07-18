# Fraud Detection for E-Commerce and Bank Transactions

Fraud detection system for Adey Innovations Inc., covering both e-commerce
transactions and bank credit card transactions.

## Business Context
Improves fraud detection accuracy while balancing false positives (customer
friction) against false negatives (financial loss), using geolocation
analysis, transaction pattern recognition, and explainable ML.

## Project Structure
- `notebooks/` — EDA, feature engineering, modeling, and SHAP explainability
- `src/` — reusable modules (data processing, feature engineering, modeling)
- `tests/` — unit tests
- `scripts/` — standalone run scripts
- `models/` — saved model artifacts

## Data
Datasets are not tracked in version control (see `.gitignore`).
Place raw files in `data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

## Setup
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Status
🚧 In progress — see notebooks for current analysis.