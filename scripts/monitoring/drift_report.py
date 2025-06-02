# Generates the drift report (inference vs training data)
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os 

EXPECTED_TRAINING_DATA_FILE = "./data-lake/training/training_features_raw_for_drift.parquet"

print(f"Attempting to load training data from: {EXPECTED_TRAINING_DATA_FILE}")
try:
    df_train = pd.read_parquet(EXPECTED_TRAINING_DATA_FILE)
    print(f"Successfully loaded training data from {EXPECTED_TRAINING_DATA_FILE}, shape: {df_train.shape}")
except FileNotFoundError:
    print(f"ERROR: Training data file '{EXPECTED_TRAINING_DATA_FILE}' not found.")
    print("Please ensure 'scripts/training/save_training_data.py' has been run successfully and created this file.")
    print("This specific file should contain unscaled numerical features and raw string categoricals for proper drift comparison.")
    exit() # Exit if the crucial training baseline is missing

df_train["dataset"] = "train"

inference_files = glob.glob("./data-lake/inference/*.parquet")
if not inference_files:
    print("No inference data found to compare. Skipping drift report.")
    df_inf = pd.DataFrame() # Create an empty DataFrame if no inference files
else:
    df_inf = pd.concat([pd.read_parquet(f) for f in inference_files], ignore_index=True)
    print(f"Loaded {len(inference_files)} inference files. Total inference samples: {len(df_inf)}")

# Add dataset column to inference data only if it's not empty
if not df_inf.empty:
    df_inf["dataset"] = "inference"

# Combine
# Handle case where df_inf might be empty
if not df_inf.empty:
    df_all = pd.concat([df_train, df_inf], ignore_index=True)
else:
    print("Inference data is empty, proceeding with training data only for any potential individual analysis (drift comparison will be N/A).")
    df_all = df_train.copy() # Or handle as an error if both are needed for all subsequent steps

# Plot and KS-test
# Ensure 'dataset' column exists if df_all was just df_train
if 'dataset' not in df_all.columns:
    print("Error: 'dataset' column missing in df_all. This shouldn't happen if df_train was loaded.")
    exit()

# Identify features - ensure columns exist after potential empty df_inf handling
features_to_compare = [col for col in df_train.columns if col.startswith("feature_") or col == "age"]
# For categorical features, we need to ensure they are present in both if we compare later
# This script primarily focuses on numerical 'feature_' and 'age' based on original structure

print(f"\nComparing distributions for features: {features_to_compare}")

if 'inference' not in df_all['dataset'].unique() and not df_inf.empty:
    print("Warning: Inference data was loaded but 'inference' source_type is not in the combined data. Check 'dataset' column assignment.")
elif df_inf.empty:
    print("No inference data loaded. KS tests and comparative density plots will be skipped.")

for feature in features_to_compare:
    if feature not in df_all.columns:
        print(f"Warning: Feature '{feature}' not found in combined data. Skipping.")
        continue

    print(f"\n--- Analyzing Feature: {feature} ---")
    
    # Density Plot
    plt.figure(figsize=(10, 6)) # Create a new figure for each plot
    if not df_inf.empty: # Only plot grouped if inference data exists
        df_all.groupby("dataset")[feature].plot(kind='density', legend=True, title=f"Distribution of {feature} (Training vs. Inference)")
    else: # Plot only training data if no inference data
        df_train[feature].plot(kind='density', label='training', legend=True, title=f"Distribution of {feature} (Training Data Only)")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # KS test - only if inference data is present and feature exists in both
    if not df_inf.empty and feature in df_train.columns and feature in df_inf.columns:
        train_feature_data = df_train[feature].dropna()
        inference_feature_data = df_inf[feature].dropna()

        if len(train_feature_data) > 1 and len(inference_feature_data) > 1: # KS test needs some data
            stat, p = ks_2samp(train_feature_data, inference_feature_data)
            print(f"  {feature}: KS Stat={stat:.4f}, p-value={p:.4f}")
            if p < 0.05:
                print(f"ALERT: Significant difference detected for {feature} (p < 0.05)")
        else:
            print(f"Skipping KS test for {feature} due to insufficient data in one or both datasets after dropping NaNs.")
    elif not df_inf.empty:
        print(f"Skipping KS test for {feature} as it might be missing in training or inference data after processing.")