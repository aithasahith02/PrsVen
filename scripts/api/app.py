# Deploying RSF (outperformed) model using FastAPI 
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and previously saved preprocessors (Scaler and Label Encoder)
rsf_model = joblib.load("models/rsf_model.pkl")
scaler = joblib.load("models/scaler.joblib")
encoder = joblib.load("models/encoder.joblib")  

# FastAPI app
app = FastAPI(title="Survival Prediction API")

# Define input schema
class PatientData(BaseModel):
    gender: str
    smoker_status: str
    age: float
    features: list[float]  # All the 75 features

@app.post("/predict")
def predict_survival(data: PatientData):
    input_dict = {
        "gender": [data.gender],
        "smoker_status": [data.smoker_status],
        "age": [data.age]
    }
    for i, val in enumerate(data.features):
        input_dict[f"feature_{i+1}"] = [val]
    
    df = pd.DataFrame(input_dict)

    df["gender"] = encoder.transform(df["gender"])
    df["smoker_status"] = encoder.transform(df["smoker_status"])

    X = scaler.transform(df)

    # Predicting using the survival function
    surv_funcs = rsf_model.predict_survival_function(X)
    times = surv_funcs[0].x
    survival_probs = surv_funcs[0].y

    # Compute mean life expectancy
    expected_months = float(np.trapz(survival_probs, times))

    return {
        "survival_curve": list(zip(times.tolist(), survival_probs.tolist())),
        "expected_survival_months": round(expected_months, 2)
    }