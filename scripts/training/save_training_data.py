import pandas as pd
import os
from datetime import datetime

DATA_LAKE_BASE_DIR = "./data-lake"
TRAINING_DATA_DIR = os.path.join(DATA_LAKE_BASE_DIR, "training")
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

# Saving data with time stamp

training_data_filename = "2024-12-01_training_data.parquet" 
try:
    df = pd.read_csv("../../data/processed/cleaned_data.csv")
except FileNotFoundError:
    print("Error: ../../data/processed/cleaned_data.csv not found.")
    # Cleaned data from the root
    try:
        df = pd.read_csv("data/processed/cleaned_data.csv")
        print("Data Loaded from cleaned_csv")
    except FileNotFoundError:
        print("Could not find cleaned_data.csv at 'data/processed/cleaned_data.csv'")
        exit()

# Save to Parquet
output_path = os.path.join(TRAINING_DATA_DIR, training_data_filename)
df.to_parquet(output_path, index=False)

print(f"Training data saved to: {output_path}")