# Evaluation entry point. Loads each trained artifact and the saved test split,
# then reports metrics and writes plots to reports/. Run `python src/train.py`
# first to produce the artifacts.
import os
import sys

import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import config
from evaluate import (
    evaluate_model,
    plot_roc_auc_curve,
    plot_feature_importance,
    tune_threshold,
)


def load_test_split():
    path = os.path.join(config.MODELS_DIR, "test_split.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test split not found at {path}. Run `python src/train.py` first."
        )
    data = joblib.load(path)
    return data["X_test"], data["y_test"]


def main() -> None:
    X_test, y_test = load_test_split()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    for name in config.build_models():
        artifact = config.artifact_path(name)
        if not os.path.exists(artifact):
            print(f"[skip] {name}: artifact not found ({artifact})")
            continue

        print(f"\n{'=' * 60}\nEvaluating {name}\n{'=' * 60}")
        pipe = joblib.load(artifact)

        evaluate_model(pipe, X_test, y_test, threshold=config.DECISION_THRESHOLD)
        best_threshold = tune_threshold(pipe, X_test, y_test, name=name)
        print(f"Suggested threshold (max F1): {best_threshold:.3f}")

        plot_roc_auc_curve(pipe, X_test, y_test, name=name)
        plot_feature_importance(pipe, name=name)

    print(f"\nDone. Plots written to {config.REPORTS_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
