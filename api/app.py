import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Make the shared config importable regardless of where streamlit is launched.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("Churn Prediction")

FEATURE_ORDER = [
    "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Subscription Type", "Contract Length",
    "Total Spend", "Last Interaction",
]


@st.cache_resource
def load_model(name: str):
    """Load a trained pipeline, with a clear error if the artifact is missing."""
    path = config.artifact_path(name)
    if not os.path.exists(path):
        st.error(
            f"Model artifact not found: `{path}`.\n\n"
            "Train the models first by running `python src/train.py`."
        )
        st.stop()
    return joblib.load(path)


def risk_label(prob: float):
    """Map a churn probability to a (renderer, message) pair using config bands."""
    if prob < config.RISK_BANDS["low"]:
        return st.success, "Low risk — maintain current engagement strategy."
    if prob < config.RISK_BANDS["medium"]:
        return st.warning, "Medium risk — consider targeted retention campaigns."
    return st.error, "High risk — immediate retention action recommended."


# --- Sidebar: model selection ---------------------------------------------
available = [m for m in config.build_models() if os.path.exists(config.artifact_path(m))]
if not available:
    st.error("No trained models found. Run `python src/train.py` first.")
    st.stop()

model_name = st.sidebar.selectbox(
    "Model", available,
    index=available.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in available else 0,
)
threshold = st.sidebar.slider(
    "Decision threshold", 0.0, 1.0, float(config.DECISION_THRESHOLD), 0.01,
    help="Probability at or above which a customer is flagged as churn.",
)
model = load_model(model_name)

tab_single, tab_batch = st.tabs(["Single customer", "Batch scoring (CSV)"])

# --- Single prediction -----------------------------------------------------
with tab_single:
    st.write("Enter customer information to predict churn risk.")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)
        usage_frequency = st.number_input("Usage Frequency", min_value=0, value=10)
        support_calls = st.number_input("Support Calls", min_value=0, value=1)
        payment_delay = st.number_input("Payment Delay (days)", min_value=0, value=0)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Total Spend", min_value=0.0, value=1000.0)
        last_interaction = st.number_input("Days Since Last Interaction", min_value=0, value=5)

    if st.button("Predict Churn"):
        input_df = pd.DataFrame([{
            "Age": age, "Gender": gender, "Tenure": tenure,
            "Usage Frequency": usage_frequency, "Support Calls": support_calls,
            "Payment Delay": payment_delay, "Subscription Type": subscription_type,
            "Contract Length": contract_length, "Total Spend": total_spend,
            "Last Interaction": last_interaction,
        }])[FEATURE_ORDER]

        try:
            probability = float(model.predict_proba(input_df)[0][1])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        st.subheader("Churn Risk Assessment")
        flagged = "CHURN" if probability >= threshold else "RETAIN"
        render, message = risk_label(probability)
        render(f"{flagged} — churn probability {probability:.2%}")
        st.write(message)
        st.caption(f"Model: {model_name} · threshold: {threshold:.2f}")

# --- Batch scoring ---------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV with the customer feature columns to score in bulk.")
    st.caption("Required columns: " + ", ".join(FEATURE_ORDER))
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        missing = [c for c in FEATURE_ORDER if c not in batch.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        try:
            probs = model.predict_proba(batch[FEATURE_ORDER])[:, 1]
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")
            st.stop()

        result = batch.copy()
        result["churn_probability"] = probs
        result["prediction"] = (probs >= threshold).astype(int)
        st.dataframe(result)
        st.metric("Flagged as churn", f"{int(result['prediction'].sum())} / {len(result)}")
        st.download_button(
            "Download scored CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name="churn_scored.csv",
            mime="text/csv",
        )
