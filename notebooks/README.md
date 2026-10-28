# Notebooks

Execution order:
1. `eda-fraud-data.ipynb` — cleaning + EDA for Fraud_Data.csv
2. `eda-creditcard.ipynb` — cleaning + EDA for creditcard.csv
3. `feature-engineering-fraudData.ipynb` — geolocation merge, feature
   engineering, transformation, and SMOTE for Fraud_Data.csv
4. `feature-engineering-creditcard.ipynb` — feature engineering,
   transformation, and SMOTE for creditcard.csv
5. `modeling_creditcard.ipynb` — baseline vs ensemble models, evaluation, model selection
6. `modeling_fraud.ipynb` - baseline vs ensemble mdoels, evaluation, model selection
6. `shap-explainability.ipynb` — SHAP analysis and business recommendations


## Task 1: key data findings
 
- **E-commerce:** `purchase_value` and `age` showed no real fraud signal, and `user_transaction_count` was constant at 1.
- `device_shared_count` was the strongest engineered signal: the fraud rate climbs from about 3% at a count of 1 to about 90% at a count of 10.
- `time_since_signup` is bimodal. The fraud median is about 1 second after signup, against about 60 days for legitimate users.
- Fraud rates were elevated for Ecuador, Tunisia and Peru. IPs with unknown country (about 14.5% of rows) were slightly below the baseline fraud rate.
- **Credit card:** `Amount` was heavily right-skewed and log-transformed. V1-V28 are PCA components and are treated as uninterpretable by design.

## Task 2: modeling approach
 
- **Models:** Logistic Regression (baseline) and LightGBM (ensemble), per dataset.
- **Imbalance handling:** two routes, compared directly: SMOTE on the training data only, and class weighting on the original training data. The test set is never resampled.
- **Evaluation:** one shared `evaluate_model()` function for every model and both datasets, so numbers are comparable. All results below are on the held-out test set at a 0.5 threshold, except PR-AUC and ROC-AUC, which are threshold-free.
## Results
 
### E-commerce (`Fraud_Data.csv`)
 
| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LightGBM (class-weight) | 0.7072 | 0.8372 | 0.5434 | 0.6943 | 0.6097 |
| LogReg (class-weight) | 0.6604 | 0.8358 | 0.6193 | 0.6346 | 0.6269 |
| LightGBM (SMOTE)* | 0.7080 | 0.8397 | 0.9740 | 0.5297 | 0.6862 |
| LogReg (SMOTE)* | 0.6307 | 0.8117 | 0.7508 | 0.5781 | 0.6532 |
 
\* Trained on a SMOTE set containing invalid synthetic rows (see Limitations). Kept as a documented experiment, not as primary results.
 
### Credit card (`creditcard.csv`)
 
| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LightGBM (SMOTE) | 0.8144 | 0.9655 | 0.8000 | 0.8000 | 0.8000 |
| LogReg (SMOTE) | 0.6911 | 0.9635 | 0.1341 | 0.8526 | 0.2318 |
| LightGBM (class-weight, sqrt ratio) | 0.3586 | 0.8678 | 0.4861 | 0.7368 | 0.5858 |
| LightGBM (class-weight, full ratio) | 0.0812 | 0.8579 | 0.1026 | 0.7368 | 0.1802 |
 
## Findings
 
**E-commerce**
- LightGBM leads the clean logistic regression by about 0.05 PR-AUC (0.707 vs 0.660). The lead is moderate, not dominant, and ROC-AUC is essentially tied.
- SMOTE gave LightGBM no PR-AUC benefit over class weighting (0.708 vs 0.707) and hurt Logistic Regression (0.631 vs 0.660). The different precision and recall of the LightGBM arms reflect where the 0.5 threshold lands on each model's probability scale, not different ranking quality.
- **About half of the fraud is invisible to the current features.** Fraud caught by LightGBM has extreme `device_shared_count` and a signup-to-purchase time of seconds. Missed fraud matches legitimate transactions on every feature. The ceiling is feature-limited, not model-limited.
- Logistic Regression assigns almost no weight to `time_since_signup_hours` (a near-zero coefficient) despite it being a strong signal, because fraud sits at a cliff at the low extreme while legitimate users spread across the whole range. A straight line cannot represent that; tree splits can.
**Credit card**
- LightGBM with SMOTE is the clear winner on PR-AUC (0.814): 76 of 95 frauds caught with 19 false alarms.
- ROC-AUC is almost identical for LightGBM and Logistic Regression (0.9655 vs 0.9635), yet at 0.5 the logistic regression raises 523 false alarms to catch 81 frauds. PR-AUC exposes a difference ROC-AUC hides.
- Class weighting performed poorly as configured, and the full-ratio arm is unstable (PR-AUC measured 0.057 in one run and 0.081 in another with identical settings). With only about 380 real fraud rows in training, heavily weighted trees overfit them. SMOTE appears to give the trees a smoother fraud region to learn, despite over 99.8% of the resampled fraud class being synthetic.
## Recommendation
 
| Dataset | Recommended model | Why |
|---|---|---|
| E-commerce | LightGBM, class-weight | Best clean PR-AUC; trained on valid data. Final threshold should be set from the cost of missed fraud versus false alarms (not yet done). |
| Credit card | LightGBM, SMOTE | Best PR-AUC by a wide margin with balanced precision and recall at 0.5. |
 
A cost-sensitivity check in `model_comparison.ipynb` (assumed cost ratios, default 0.5 threshold) shows that on the e-commerce data LightGBM is cheaper than the class-weighted logistic regression once a missed fraud costs more than about 3 false alarms. On the credit card data, LightGBM (SMOTE) stays cheapest until a missed fraud costs about 100 false alarms, where the logistic regression's higher recall catches up. These ratios are assumptions, not measured business costs.
 
If explainability is required, Logistic Regression on the e-commerce data is the interpretable fallback (its coefficients are meaningful, for example `device_shared_count`) at a cost of about 0.05 PR-AUC. On the credit card data the V1-V28 features are anonymized, so the interpretability argument for Logistic Regression is weak. Model explanations for the recommended models are planned in Task 3 (SHAP).
 
## Methodological notes and limitations
 
These issues were found during modeling and are documented rather than hidden.
 
1. **Invalid SMOTE rows in the e-commerce data.** SMOTE was applied directly to one-hot columns and the result was cast back to booleans. About 15,578 resampled rows carry two countries at once, plus 1,269 with two browsers and 474 with two sources, which is impossible for a real transaction. The SMOTE arms inflated dummy-variable coefficients and are not used as primary results. Class weighting is primary for this dataset. A proper fix (SMOTE-NC before one-hot encoding) was not implemented.
2. **Scale mismatch in the saved credit card arrays.** The saved resampled training arrays had `Time`, `hour_of_day` and `Amount_log` standardized while the saved train and test arrays were raw. Models trained on the resampled array and tested on the raw test set produced invalid results: logistic regression predicted no fraud at all (PR-AUC 0.002), and LightGBM scored 0.702 instead of 0.814. The modeling notebook now rebuilds SMOTE from the raw training arrays, and the saved resampled pickle is not used for credit card modeling.
3. **Class weights must be scaled to the imbalance.** The same `scale_pos_weight` strategy that worked at a 1:10 imbalance (e-commerce) failed at 1:600 (credit card). A square-root heuristic was chosen in advance, not tuned on the test set, and it still underperformed.
4. **Single train/test split, no confidence intervals.** With only 95 fraud cases in the credit card test set, one fraud shifts recall by about 1 percentage point. Differences between close models should not be over-read.
5. **Threshold fixed at 0.5 and no hyperparameter tuning.** Operating points are not tuned to a cost model. Any tuning should use a validation split from the training data, never the test set.
6. **Not run:** a class-weight Logistic Regression arm on the credit card data, so the credit card baseline was only trained with SMOTE.
## Next steps
 
- Stratified k-fold cross-validation (mean and std) and light hyperparameter tuning.
- Threshold selection from an explicit cost ratio, on a validation split.
- Bootstrap confidence intervals for PR-AUC.
- Task 3: SHAP explanations for the recommended models.
- E-commerce feature ideas: number of users per IP address, short signup-to-purchase flags, and user-level aggregates.
- Proper SMOTE-NC for the e-commerce data to settle the SMOTE-versus-class-weight question.