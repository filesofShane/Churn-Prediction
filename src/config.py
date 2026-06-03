# Central configuration for the churn-prediction pipeline.
# Everything that is likely to drift between training, evaluation and the
# dashboard (paths, artifact names, the model registry and the decision
# threshold) lives here so there is a single source of truth.
import os

# --- Project paths (absolute, so the code works regardless of cwd) ---------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# --- Data ------------------------------------------------------------------
KAGGLE_DATASET = "muhammadshahidazeem/customer-churn-dataset"
TRAIN_FILE = "customer_churn_dataset-training-master.csv"
TEST_FILE = "customer_churn_dataset-testing-master.csv"

TARGET = "Churn"
ID_HINT = "id"          # feature columns containing this substring are dropped
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Artifacts -------------------------------------------------------------
ARTIFACT_TMPL = "churn_model_{name}.pkl"


def artifact_path(name: str) -> str:
    """Absolute path to a trained model artifact for the given model name."""
    return os.path.join(MODELS_DIR, ARTIFACT_TMPL.format(name=name))


# --- Model registry --------------------------------------------------------
# The dashboard loads the DEFAULT_MODEL artifact, so add a new entry here and
# retrain to make it available for decisioning. The keys are used as artifact
# names and in the evaluation report, so choose them to be descriptive but concise.
def build_models() -> dict:
    """Return a fresh {name: estimator} mapping. One artifact is saved per key."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    return {
        "LogReg": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        # I capped depth/leaf size because the unbounded forest grew to 1.6 GB;
        # this shrinks it drastically with negligible metric loss on this data.
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradBoost": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


# --- Decisioning / dashboard ----------------------------------------------
# DEFAULT_MODEL is the artifact the dashboard loads.
# DECISION_THRESHOLD is the operating point chosen during evaluation; update it
# here after reviewing the precision/recall tradeoff (evaluate.tune_threshold).
DEFAULT_MODEL = "RandomForest"
DECISION_THRESHOLD = 0.50

# P(churn) < LOW -> low risk; < MEDIUM -> medium; otherwise high.
RISK_BANDS = {"low": 0.40, "medium": 0.70}
