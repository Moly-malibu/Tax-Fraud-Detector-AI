# api/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/tax_model.pkl")

class TaxInput(BaseModel):
    income: float
    deductions: float
    tax: float

@app.post("/predict")
def predict(data: TaxInput):
    df = pd.DataFrame([data.dict()])
    df["ded_ratio"] = df["deductions"] / df["income"]
    df["tax_ratio"] = df["tax"] / df["income"]
    prob = model.predict_proba(df)[0][1]
    return {"fraud_risk": prob}