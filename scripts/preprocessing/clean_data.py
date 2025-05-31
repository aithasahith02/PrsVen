import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

# Data Loading
INPUT_PATH = "data/raw/survival_data.csv"
OUTPUT_PATH = "data/processed/cleaned_data.csv"
SCALER_PATH = "models/scaler.joblib"
ENCODER_PATH = "models/encoder.joblib"

def clean_and_transform():
    df = pd.read_csv(INPUT_PATH)

    # Dates handling
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["birth_date"] = pd.to_datetime(df["birth_date"])
    df["age"] = (df["observation_date"] - df["birth_date"]).dt.days // 365

    # Label Encoding gender and 
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])
    df["smoker_status"] = le.fit_transform(df["smoker_status"])

    # Dropping original date columns
    df.drop(["birth_date", "observation_date", "patient_id"], axis=1, inplace=True)

    # Normalization of feature columns
    feature_cols = [col for col in df.columns if col.startswith("features_")] + ["age"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # saving cleaned data back to same dir
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Saving transformers
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)

    print(f"Data Cleaned and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_and_transform()
