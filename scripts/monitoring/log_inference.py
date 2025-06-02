# Logs the inferences
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Load training data
df_train = pd.read_parquet("./data-lake/training/2024-12-01_training_data.parquet")
df_train["dataset"] = "train"

# Load inference files
inference_files = glob.glob("./data-lake/inference/*.parquet")
if not inference_files:
    print("⚠️ No inference data found to compare. Skipping drift report.")
    exit(0)

df_inf = pd.concat([pd.read_parquet(f) for f in inference_files], ignore_index=True)
df_inf["dataset"] = "inference"

# Combine
df_all = pd.concat([df_train, df_inf], ignore_index=True)

# Plot and KS-test
features = [col for col in df_all.columns if col.startswith("feature_")]

for feature in features:
    df_all.groupby("dataset")[feature].plot(kind='density', legend=True, title=f"Feature Drift: {feature}")
    plt.xlabel(feature)
    plt.legend()
    plt.show()

    stat, p = ks_2samp(df_train[feature], df_inf[feature])
    print(f"{feature}: KS Stat={stat:.4f}, p-value={p:.4f}")
