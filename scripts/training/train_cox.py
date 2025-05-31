"""Cox Proportional Hazards model is chosen as first because it's fast,
 interpretable, and sets a solid baseline for survival analysis using clean, 
 structured data."""

import pandas as pd
from lifelines import CoxPHFitter
import joblib
import os

# Data files
DATA_PATH = "data/processed/cleaned_data.csv"
MODEL_PATH = "models/cox_model.pkl"

def train_cox_model():
    # Load cleaned data
    df = pd.read_csv(DATA_PATH)

    # Prepare survival data
    duration_col = "time_to_event"
    event_col = "event"

    # Train Cox model
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(cph, MODEL_PATH)

    print(f"First model - Cox trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_cox_model()

