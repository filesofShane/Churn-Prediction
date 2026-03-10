import joblib
import pandas as pd
import streamlit as st

# Loading the trained model
model = joblib.load("models/churn_model_LogReg.pkl")

st.title("Churn Prediction")

st.write("Enter customer information to predict churn risk.")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.number_input("Tenure (months)", min_value=0, value=12)

usage_frequency = st.number_input("Usage Frequency", min_value=0, value=10)

support_calls = st.number_input("Support Calls", min_value=0, value=1)

payment_delay = st.number_input("Payment Delay (days)", min_value=0, value=0)

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

contract_length = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Annual"]
)

total_spend = st.number_input("Total Spend", min_value=0.0, value=1000.0)

last_interaction = st.number_input("Days Since Last Interaction", min_value=0, value=5)

# Prediction button
if st.button("Predict Churn"):

    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_type],
        "Contract Length": [contract_length],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Churn Risk Assessment")

    if probability < 0.40:
        st.success(f" Low Risk: {probability:.2%}")
        st.write("Customer is unlikely to churn. Maintain current engagement strategy.")
    elif probability < 0.70:
        st.warning(f" Medium Risk: {probability:.2%}")
        st.write("Customer shows signs of churn risk. Consider targeted retention campaigns.")
    else:
        st.error(f" High Risk: {probability:.2%}")
        st.write("Customer is highly likely to churn. Immediate retention action recommended.")