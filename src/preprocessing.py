# Data cleaning and the modelling preprocessor.
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def clean_df(df: pd.DataFrame, target: str = config.TARGET) -> pd.DataFrame:
    """Light cleaning only. Feature-level missing values are handled by the
    preprocessor's imputers, so we do NOT drop rows on feature nulls here."""
    df = df.copy()

    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found. Available: {list(df.columns)}"
        )

    # Drop rows with missing target and convert to int (the source data has some nulls and the type is object).
    df = df[df[target].notna()]
    df[target] = df[target].astype(int)

    # Strip whitespace from string columns to avoid issues with unseen categories during inference.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def split_xy(df: pd.DataFrame, target: str = config.TARGET):
    """Separate features/target and drop identifier-like columns."""
    X = df.drop(columns=[target])
    y = df[target]

    drop_cols = [c for c in X.columns if config.ID_HINT in c.lower()]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numeric: median-impute + scale (needed for LogReg convergence and
    interpretable coefficients). Categorical: mode-impute + one-hot."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
