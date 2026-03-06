## Customer Churn Prediction

## Problem

Customer churn significantly impacts recurring revenue.
This project builds a machine learning pipeline to predict high-risk customers and simulate retnetion targeting strategies

## Dataset
- 440,00+ customer records
- Features include: Tenure, Usage Frequency, Support Calls, Payment Delay, etc.
- Bianry target variable: Churn (1 = churned, 0 = retained)

## Project Structure
│
├── data/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│
├── main.py
├── requirements.txt
└── README.md

## Methodology

The project follows a standard machine learning pipeline:
1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Preprocessing
5. Model Training
6. Model Evaluation
7. Model deployment via FastAPI

Model comparison in progress. Planned models include:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting

Evaluation will focus on:
- ROC-AUC
- Precision / Recall
- Confusion Matrix

## Results
To be finalised after model benchmarking and evaluation

## Business Implications
Insights will be dreived after selecting the optimal recall-precisin balance for churn detection.

## How to Run
1. Create virtual environment
python -m venv venv

2. Activate environment
source venv/bin/activate or Windows equivalent

3. Install dependencies
pip install -r requirements.txt

4. Run the pipeline
python main.py

## Model Performance thus far

### ROC Curve
<img src="reports/ROC_AUC plot.png" width="600">

The model achieved an ROC-AUC score of **0.91**, indicating strong classification performance.

### Classification Report
<img src="reports/Classification Report.png" width="600">

### Confusion Matrix
<img src="reports/Confusion Matrix.png" width="600">

