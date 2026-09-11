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

## Key Findings (Task 1)

**Fraud_Data.csv** (9.36% fraud rate):
- `device_shared_count` is the strongest engineered signal: fraud rate
  climbs from ~3% (device used by 1 user) to ~90% (device shared across
  10+ users) — a device tied to many accounts is a strong fraud indicator.
- `time_since_signup` shows a stark median gap: fraudulent purchases occur
  a median of ~1 second after signup, versus ~60 days for legitimate users.
- Geolocation reveals elevated fraud rates in specific countries (e.g.
  Ecuador, Tunisia, Peru at 2.5–3x the baseline rate), likely reflecting
  proxy/VPN use or account takeover rather than any property of the
  country itself.
- `purchase_value` and `age` showed no meaningful difference between
  fraud and legitimate transactions — a notable negative result.

**creditcard.csv** (0.17% fraud rate, extreme imbalance):
- `Amount` is heavily right-skewed; addressed with a log transform.
- Fraud transactions show a bimodal amount pattern: lower median amount
  than legitimate transactions, but higher mean — a mix of small "test"
  charges and rare large-value fraud.
- V1–V28 are PCA-transformed and not individually interpretable by
  design (privacy protection); interpretation relies on SHAP rather
  than raw feature meaning (see Task 3).
- Given ~0.17% fraud prevalence, SMOTE-resampled training data is
  >99.8% synthetic fraud examples — a known limitation of oversampling
  under extreme imbalance, documented rather than hidden.

## Status
✅ Task 1 (Data Analysis & Preprocessing) — complete
🚧 Task 2 (Model Building & Training) — in progress
⬜ Task 3 (Model Explainability) — not started