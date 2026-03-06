# Importing necessary libraries
import kagglehub 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Importing necessary functions from preprocessing
from preprocessing import clean_df, split_xy, build_preprocessor

path = kagglehub.dataset_download("muhammadshahidazeem/customer-churn-dataset")

train = pd.read_csv(f"{path}/customer_churn_dataset-training-master.csv")
test  = pd.read_csv(f"{path}/customer_churn_dataset-testing-master.csv")

# Combining train and test datasets for preprocessing
df = pd.concat([train, test], ignore_index=True)

df = clean_df(df)
X, y = split_xy(df) # Splitting features and target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Building the preprocessing pipeline
prep = build_preprocessor(X_train)

model = LogisticRegression(max_iter=2000)

pipe = Pipeline([
    ("preprocessor", prep),
    ("model", model)
])

pipe.fit(X_train, y_train)

# Saving the trained model and test data for evaluation in main.py
import joblib
joblib.dump(pipe, "models/churn_model_pipeline.joblib")
print("Model training completed and saved to 'models/churn_model_pipeline.joblib'.")