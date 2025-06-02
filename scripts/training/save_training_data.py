# Saves the trainng data for drift detection
import pandas as pd
import os
from datetime import datetime 

DATA_LAKE_BASE_DIR = "./data-lake"
TRAINING_DATA_DIR = os.path.join(DATA_LAKE_BASE_DIR, "training")
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

RAW_DATA_PATH = "data/raw/survival_data.csv" 

TRAINING_SNAPSHOT_FILENAME = "training_features_raw_for_drift.parquet"

def create_training_snapshot_for_drift_comparison():
    print(f"Loading original raw training data from: {RAW_DATA_PATH}")
    try:
        df_original_raw = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Original raw data file not found at '{RAW_DATA_PATH}'.")
        print(f"Current working directory: {os.getcwd()}")
        print("Please ensure this script is run from your project root (PrsVen) and the path to survival_data.csv is correct.")
        return
    except Exception as e:
        print(f"Error loading raw data file '{RAW_DATA_PATH}': {e}")
        return

    print(f"Original raw data shape: {df_original_raw.shape}")

    df_processed = df_original_raw.copy() 
    try:
        df_processed["observation_date"] = pd.to_datetime(df_processed["observation_date"])
        df_processed["birth_date"] = pd.to_datetime(df_processed["birth_date"])
        df_processed["age"] = (df_processed["observation_date"] - df_processed["birth_date"]).dt.days // 365
    except KeyError as e:
        print(f"Error: Missing expected date columns ('observation_date', 'birth_date') for age calculation in '{RAW_DATA_PATH}': {e}")
        return
    except Exception as e:
        print(f"Error during date conversion or age calculation: {e}")
        return

    numerical_feature_names = [f'feature_{i}' for i in range(1, 76)] # feature_1 to feature_75
    columns_for_drift_comparison = ['gender', 'smoker_status', 'age'] + numerical_feature_names
    
    missing_cols = [col for col in columns_for_drift_comparison if col not in df_processed.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing after initial processing from '{RAW_DATA_PATH}': {missing_cols}")
        print(f"Available columns after age calculation: {df_processed.columns.tolist()}")
        return
        
    df_snapshot = df_processed[columns_for_drift_comparison].copy()

    try:
        df_snapshot['gender'] = df_snapshot['gender'].astype(str)
        df_snapshot['smoker_status'] = df_snapshot['smoker_status'].astype(str)
        df_snapshot['age'] = df_snapshot['age'].astype(float) # Age as float (unscaled)
        for col in numerical_feature_names:
            df_snapshot[col] = df_snapshot[col].astype(float) # Other features as float (unscaled)
    except Exception as e:
        print(f"Error during data type conversion for snapshot: {e}")
        return

    # Save to Parquet
    output_path = os.path.join(TRAINING_DATA_DIR, TRAINING_SNAPSHOT_FILENAME)
    try:
        df_snapshot.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"Training data snapshot for drift comparison saved to: {output_path}")
        print(f"Shape of saved data: {df_snapshot.shape}")
        print("This data contains raw string categoricals and unscaled numerical features (including age).")
    except Exception as e:
        print(f"Error saving training snapshot to Parquet at {output_path}: {e}")

if __name__ == "__main__":
    create_training_snapshot_for_drift_comparison()