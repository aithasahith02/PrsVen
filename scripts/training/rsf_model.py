"""
Random Survival Forest is chosen as second model due to its non-linearity and balance with performance
"""

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# Data and model files
DATA_PATH = "data/processed/cleaned_data.csv"
MODEL_PATH = "models/rsf_model.pkl"


def train_rsf():
    # Load cleaned data
    df = pd.read_csv(DATA_PATH)

    # Format survival target
    y = Surv.from_dataframe("event", "time_to_event", df)
    X = df.drop(columns=["time_to_event", "event"])

    # Train-test split with 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Defining RSF model
    model = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Saving the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f" Second model - RSF trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_rsf()
