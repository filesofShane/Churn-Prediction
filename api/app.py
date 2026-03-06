import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("models/churn_model.joblib")

@app.post("/predict")
def predict_churn(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"churn_prediction": int(prediction[0])}
