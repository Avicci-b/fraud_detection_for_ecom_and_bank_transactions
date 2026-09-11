# Notebooks

Execution order:
1. `eda-fraud-data.ipynb` — cleaning + EDA for Fraud_Data.csv
2. `eda-creditcard.ipynb` — cleaning + EDA for creditcard.csv
3. `feature-engineering-fraudData.ipynb` — geolocation merge, feature
   engineering, transformation, and SMOTE for Fraud_Data.csv
4. `feature-engineering-creditcard.ipynb` — feature engineering,
   transformation, and SMOTE for creditcard.csv
5. `modeling.ipynb` — baseline vs ensemble models, evaluation, model selection
6. `shap-explainability.ipynb` — SHAP analysis and business recommendations

Note: Feature engineering is split into two notebooks (one per dataset)
rather than one combined notebook, to keep memory usage manageable when
running SMOTE on both datasets in the same environment.