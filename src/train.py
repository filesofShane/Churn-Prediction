# Training entry point: loads data, builds one pipeline per model in the
# registry and saves each artifact. Importing this module has no side effects;
# training only runs under `if __name__ == "__main__"`.
import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import config
from preprocessing import clean_df, split_xy, build_preprocessor


def load_data() -> pd.DataFrame:
    """Download the dataset from Kaggle and return the combined frame."""
    try:
        import kagglehub
        path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    except Exception as exc:  # network / auth / missing dependency
        raise RuntimeError(
            f"Failed to download dataset '{config.KAGGLE_DATASET}'. "
            f"Check your Kaggle credentials and network connection. ({exc})"
        ) from exc

    try:
        train = pd.read_csv(os.path.join(path, config.TRAIN_FILE))
        test = pd.read_csv(os.path.join(path, config.TEST_FILE))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Expected CSV files not found under {path}. "
            "The dataset layout may have changed."
        ) from exc

    # The dataset ships pre-split; we recombine and resplit ourselves so the
    # train/test boundary is reproducible from a single random_state.
    return pd.concat([train, test], ignore_index=True)


def build_pipeline(estimator, X: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", estimator),
    ])


def train_all(X_train, y_train) -> dict:
    """Fit and persist one pipeline per registered model. Returns {name: pipe}."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    trained = {}
    for name, estimator in config.build_models().items():
        print(f"Training {name}...")
        pipe = build_pipeline(estimator, X_train).fit(X_train, y_train)
        out = config.artifact_path(name)
        joblib.dump(pipe, out)
        print(f"  saved -> {out}")
        trained[name] = pipe
    return trained


def main() -> None:
    df = clean_df(load_data())
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )

    train_all(X_train, y_train)

    # Persist the test split so main.py can evaluate without re-downloading.
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        os.path.join(config.MODELS_DIR, "test_split.pkl"),
    )
    print("Training complete. Test split saved for evaluation.")


if __name__ == "__main__":
    sys.exit(main())
