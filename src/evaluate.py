# Model evaluation: metrics, threshold tuning and diagnostic plots.
# Plots are written to config.REPORTS_DIR rather than shown, so `python main.py`
# regenerates the report images reproducibly.
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; we save instead of show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
)

import config


def _save(fig_name: str) -> None:
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    out = os.path.join(config.REPORTS_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  plot -> {out}")


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Report metrics at the given decision threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"Classification Report (threshold={threshold:.2f}):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")

    return {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc_score": auc_score,
        "threshold": threshold,
    }


def tune_threshold(model, X_test, y_test, name: str = "model") -> float:
    """Find the threshold that maximises F1 and save the PR-vs-threshold curve.
    Returns the suggested threshold for the recall/precision tradeoff."""
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # precision/recall have one more element than thresholds; align them.
    p, r, t = precision[:-1], recall[:-1], thresholds
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best_idx = int(np.argmax(f1))
    best_threshold = float(t[best_idx])

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    plt.plot(t, p, label="Precision")
    plt.plot(t, r, label="Recall")
    plt.plot(t, f1, label="F1", linestyle="--")
    plt.axvline(best_threshold, color="grey", linestyle=":",
                label=f"Best F1 @ {best_threshold:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title(f"Precision / Recall / F1 vs Threshold ({name})")
    plt.legend(loc="best")
    _save(f"Threshold Tuning ({name}).png")

    return best_threshold


def plot_roc_auc_curve(model, X_test, y_test, name: str = "model") -> None:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({name})")
    plt.legend(loc="lower right")
    _save(f"ROC ({name}).png")


def _importance_series(pipe) -> pd.Series:
    """Return a feature-importance Series that works for both linear models
    (coef_) and tree ensembles (feature_importances_)."""
    model = pipe.named_steps["model"]
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(model, "coef_"):
        values = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        raise AttributeError(
            f"{type(model).__name__} exposes neither coef_ nor feature_importances_."
        )
    return pd.Series(values, index=feature_names)


def plot_feature_importance(pipe, name: str = "model", top_n: int = 10) -> None:
    importance = _importance_series(pipe)
    # Rank by magnitude so negative coefficients are not hidden at the bottom.
    top = importance.reindex(importance.abs().sort_values().index).tail(top_n)

    plt.figure(figsize=(10, 6))
    top.plot(kind="barh")
    plt.title(f"Top {top_n} Feature Importances ({name})")
    plt.xlabel("Importance / Coefficient")
    _save(f"Feature Importance ({name}).png")
