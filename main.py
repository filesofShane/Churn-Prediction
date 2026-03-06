# Main execution file for the churn prediction model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from train import pipe, X_test, y_test
from evaluate import evaluate_model, plot_roc_auc_curve, plot_feature_importance

if __name__ == "__main__":
    print("Evaluating the model on the test set...")
    evaluation_results = evaluate_model(pipe, X_test, y_test)
    print("Evaluation completed.")

    # Plot ROC-AUC curve
    plot_roc_auc_curve(pipe, X_test, y_test)

    # Plot feature importance
    plot_feature_importance(pipe)
    

