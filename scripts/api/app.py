from fastapi import FastAPI, HTTPException # Import HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessors
rsf_model = joblib.load("models/rsf_model.pkl")
scaler = joblib.load("models/scaler.joblib")
gender_encoder = joblib.load("models/gender_encoder.joblib")
smoker_encoder = joblib.load("models/smoker_encoder.joblib")

# Define FastAPI app
app = FastAPI(title="Survival Prediction API")

# Define input schema
class PatientData(BaseModel):
    gender: str
    smoker_status: str
    age: float
    features: list[float]

@app.post("/predict")
def predict_survival(data: PatientData):
    expected_n_features = 75 # Number of 'feature_X' columns
    if len(data.features) != expected_n_features:
        raise HTTPException(status_code=400, detail=f"Expected {expected_n_features} features, but got {len(data.features)}.")

    df_for_scaling_dict = {}
    for i, val in enumerate(data.features):
        df_for_scaling_dict[f"feature_{i+1}"] = [val] 
    df_for_scaling_dict["age"] = [data.age] 
    
    df_to_scale = pd.DataFrame(df_for_scaling_dict)
    
    # Explicitly order columns for the scaler as they were during fitting
    scaler_input_columns = [f"feature_{i+1}" for i in range(expected_n_features)] + ["age"]
    df_to_scale = df_to_scale[scaler_input_columns]

    # scaler.transform returns a NumPy array.
    scaled_features_and_age_array = scaler.transform(df_to_scale)
    
    df_scaled_numerical = pd.DataFrame(scaled_features_and_age_array, columns=scaler_input_columns)

    # Encoding
    try:
        encoded_gender = gender_encoder.transform([data.gender])[0]
        encoded_smoker_status = smoker_encoder.transform([data.smoker_status])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid categorical value provided: {str(e)}. Ensure gender is one of {gender_encoder.classes_} and smoker_status is one of {smoker_encoder.classes_}.")

    final_model_input_dict = {}
    final_model_input_dict["gender"] = [encoded_gender]
    final_model_input_dict["smoker_status"] = [encoded_smoker_status]

    for i in range(expected_n_features):
        col_name = f"feature_{i+1}"
        final_model_input_dict[col_name] = [df_scaled_numerical[col_name].iloc[0]]
    
    final_model_input_dict["age"] = [df_scaled_numerical["age"].iloc[0]]
        
    # Define the exact column order the RSF model expects (based on rsf_model.py)
    model_training_columns_order = ["gender", "smoker_status"] + [f"feature_{i+1}" for i in range(expected_n_features)] + ["age"]
    
    # Create the DataFrame with the specified column order
    final_model_input_df = pd.DataFrame(final_model_input_dict, columns=model_training_columns_order)

    # Predict function
    try:
        surv_funcs = rsf_model.predict_survival_function(final_model_input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")

    times = surv_funcs[0].x
    survival_probs = surv_funcs[0].y
    expected_months = float(np.trapz(survival_probs, times))

    return {
        "survival_curve": list(zip(times.tolist(), survival_probs.tolist())),
        "expected_survival_months": round(expected_months, 2)
    }