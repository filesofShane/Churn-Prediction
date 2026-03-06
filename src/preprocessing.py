# Importing necessary libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing function to clean the DataFrame
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Avoid modifying the original DataFrame

    # Drop rows with null values
    df = df.dropna()
    
    # Ensure 'Churn' column exists
    df = df[df["Churn"].notna()]
    df["Churn"] = df["Churn"].astype(int)

    # Clean all strings columns by stripping whitespace    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    return df


# Function to split features and target variable
def split_xy(df: pd.DataFrame, target: str = "Churn"):
    # Separate features and target variable
    X = df.drop(columns=[target])
    y = df[target]
    
    # Drop ID like columns if they exist
    drop_cols = [col for col in X.columns if 'id' in col.lower()]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X, y

# Building a preprocessor to ensure the data is in the correct format for modeling
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))])
    
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
    
    return ColumnTransformer(
        transformers =[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )