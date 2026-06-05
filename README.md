# Customer Churn Prediction

## Problem

Customer churn significantly impacts recurring revenue.
This project builds a machine learning pipeline to predict high-risk customers and simulate retention targeting strategies.

## Dataset
- 440,000+ customer records
- Features include: Tenure, Usage Frequency, Support Calls, Payment Delay, etc.
- Binary target variable: Churn (1 = churned, 0 = retained)

## Project Structure
- api/
  - app.py — Streamlit dashboard (single + batch scoring)
- notebooks/
  - eda.ipynb
- src/
  - config.py — single source of truth (paths, model registry, thresholds)
  - preprocessing.py
  - train.py — trains every model in the registry, saves one artifact each
  - evaluate.py
- models/
  - churn_model_LogReg.pkl
  - churn_model_RandomForest.pkl
  - churn_model_GradBoost.pkl
  - test_split.pkl — held-out split saved by training for evaluation
- main.py — loads artifacts and writes evaluation reports
- requirements.txt
- README.md

## Methodology

The project follows a standard machine learning pipeline:
1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Preprocessing (median/most-frequent imputation, scaling, one-hot encoding)
5. Model Training (config-driven; one artifact per model)
6. Model Evaluation (metrics, ROC, threshold tuning, feature importance)
7. Model deployment via Streamlit

Models are defined in `src/config.py` and trained together:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting

Evaluation focuses on:
- ROC-AUC
- Precision / Recall (with explicit decision-threshold tuning)
- Confusion Matrix

## How to Run
```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models (downloads the dataset, saves artifacts to models/)
python src/train.py

# 4. Evaluate and regenerate report plots in reports/
python main.py

# 5. Launch the dashboard
streamlit run api/app.py
```

## Results

Metrics below are for the **churn class (1)** on a held-out test set of 101,042
customers, evaluated at the default 0.50 threshold. Full output is reproducible
with `python main.py`.

| Model | ROC-AUC | Precision | Recall | F1 | Accuracy | Suggested threshold (max F1) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.909 | 0.89 | 0.82 | 0.85 | 0.84 | 0.354 |
| Random Forest | **0.953** | 0.90 | **0.98** | **0.94** | **0.93** | 0.180 |
| Gradient Boosting | 0.951 | 0.90 | 0.97 | 0.93 | 0.92 | 0.185 |

**Random Forest is the strongest model** and is the dashboard default
(`DEFAULT_MODEL` in `src/config.py`). The two tree ensembles materially
outperform the linear baseline, mainly on recall. The forest is depth-limited
(`max_depth=12`, `min_samples_leaf=50`) to keep the artifact small without
sacrificing accuracy.

> **Note on the very high Random Forest recall (845 false negatives / 56,099).**
> This dataset is quasi-synthetic and contains very strong, near-deterministic
> signals (e.g. Support Calls and Payment Delay). The headline numbers should be
> read with that in mind rather than as evidence of production-grade performance;
> on real customer data, expect lower and noisier metrics.

### Model Performance (Logistic Regression)
ROC-AUC **0.909**. Confusion matrix (rows = actual, cols = predicted):

|  | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 39,040 | 5,903 |
| **Actual 1** | 9,947 | 46,152 |

<img src="reports/ROC (LogReg).png" width="500"> <img src="reports/Threshold Tuning (LogReg).png" width="500">

### Model Performance (Random Forest)
ROC-AUC **0.953**. Confusion matrix:

|  | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 38,683 | 6,260 |
| **Actual 1** | 845 | 55,254 |

<img src="reports/ROC (RandomForest).png" width="500"> <img src="reports/Threshold Tuning (RandomForest).png" width="500">

### Model Performance (Gradient Boosting)
ROC-AUC **0.951**. Confusion matrix:

|  | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 38,800 | 6,143 |
| **Actual 1** | 1,742 | 54,357 |

<img src="reports/ROC (GradBoost).png" width="500"> <img src="reports/Threshold Tuning (GradBoost).png" width="500">

## Business Implications
The decision threshold is tuned on the precision/recall tradeoff (see the
"Threshold Tuning" plots produced by `main.py`) rather than left at the default
0.5. The chosen operating point is stored in `src/config.py` (`DECISION_THRESHOLD`)
and is used consistently by both evaluation and the dashboard, so the retention
team acts on the same risk bands the model was validated against. The max-F1
suggestions above (0.18–0.35) flag more at-risk customers than the 0.50 default,
trading a little precision for higher recall — appropriate when a missed churner
costs more than a wasted retention offer.
