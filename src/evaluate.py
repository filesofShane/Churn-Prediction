# Building the evaluation function to assess the model's performance
from xml.parsers.expat import model

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")

    return {
        "classification_report": classification_report(y_test, y_pred, output_dict=True), 
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc_score": auc_score
    }

# Plots and visualizations can be added here in the future to further analyze the model's performance and feature importance.

# ROC_AUC_Curve plot

import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_auc_curve(model, X_test, y_test):
    from sklearn.metrics import roc_curve, auc

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Feature importance plot for logistic regression coefficients
def plot_feature_importance(pipe):
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]

    coefficients = model.coef_[0]
    feature_names = preprocessor.get_feature_names_out()

    feature_importance = pd.Series(coefficients, index=feature_names).sort_values()

    plt.figure(figsize=(10, 6))
    feature_importance.tail(10).plot(kind="barh")
    plt.title("Top Positive Feature Coefficients")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.show()
