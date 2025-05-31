import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

# File paths
INPUT_PATH = "data/raw/survival_data.csv"
OUTPUT_PATH = "data/processed/cleaned_data.csv"
SCALER_PATH = "models/scaler.joblib"
GENDER_ENCODER_PATH = "models/gender_encoder.joblib"
SMOKER_ENCODER_PATH = "models/smoker_encoder.joblib"

def clean_and_transform():
    df = pd.read_csv(INPUT_PATH)

    # Handle dates
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["birth_date"] = pd.to_datetime(df["birth_date"])
    df["age"] = (df["observation_date"] - df["birth_date"]).dt.days // 365

    # Encode gender and smoker_status
    gender_encoder = LabelEncoder()
    smoker_encoder = LabelEncoder()
    df["gender"] = gender_encoder.fit_transform(df["gender"])
    df["smoker_status"] = smoker_encoder.fit_transform(df["smoker_status"])

    # Drop unnecessary columns
    df.drop(["birth_date", "observation_date", "patient_id"], axis=1, inplace=True)

    # Normalize features
    feature_cols = [col for col in df.columns if col.startswith("feature_")] + ["age"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save cleaned data
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Save transformers
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(gender_encoder, GENDER_ENCODER_PATH)
    joblib.dump(smoker_encoder, SMOKER_ENCODER_PATH)

    print(f"Data cleaned and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_and_transform()
