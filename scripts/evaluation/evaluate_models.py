"""
This is the Script to evaluate the models RSF and COX
Initial evaluation report:
"""
import pandas as pd
import joblib
import numpy as np
from lifelines.utils import concordance_index
from sksurv.util import Surv
from sksurv.metrics import (
    cumulative_dynamic_auc,
    brier_score
)

# File paths
DATA_PATH = "data/processed/cleaned_data.csv"
COX_MODEL_PATH = "models/cox_model.pkl"
RSF_MODEL_PATH = "models/rsf_model.pkl"

# Loading the data
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["time_to_event", "event"])
    y = Surv.from_dataframe("event", "time_to_event", df)
    return df, X, y

# Evaluatiing first model - COX
def evaluate_cox(df, y):
    from lifelines import CoxPHFitter

    cph = joblib.load(COX_MODEL_PATH)
    X = df.drop(columns=["time_to_event", "event"])
    times = np.array([12, 24, 36])



    surv_df = cph.predict_survival_function(X, times=times)
    surv_matrix = surv_df.T.to_numpy()
    aucs, _ = cumulative_dynamic_auc(y, y, surv_matrix, times)
    _, bs_scores = brier_score(y, y, surv_matrix, times)
    ibs = np.trapz(bs_scores, times) / (times[-1] - times[0])
    c_index = cph.concordance_index_

    return {
        "Model": "CoxPH",
        "C-Index": round(c_index, 4),
        "AUC@12": round(aucs[0], 4),
        "AUC@24": round(aucs[1], 4),
        "AUC@36": round(aucs[2], 4),
        "IBS": round(ibs, 4)
    }
# Evaluating second model - RSF
def evaluate_rsf(X, y):
    rsf = joblib.load(RSF_MODEL_PATH)
    times = np.array([12, 24, 36])
    
    surv_probs = rsf.predict_survival_function(X)
    surv_matrix = np.row_stack([fn(times) for fn in surv_probs])
    
    aucs, _ = cumulative_dynamic_auc(y, y, surv_matrix, times)
    _, bs_scores = brier_score(y, y, surv_matrix, times)
    ibs = np.trapz(bs_scores, times) / (times[-1] - times[0])
    c_index = rsf.score(X, y)

    return {
        "Model": "RSF",
        "C-Index": round(c_index, 4),
        "AUC@12": round(aucs[0], 4),
        "AUC@24": round(aucs[1], 4),
        "AUC@36": round(aucs[2], 4),
        "IBS": round(ibs, 4)
    }

# Bonus - evaluation of Mean life expectancy
def mean_life_expectancy_rsf(X, df):
    rsf = joblib.load(RSF_MODEL_PATH)
    surv_funcs = rsf.predict_survival_function(X)
    times = surv_funcs[0].x
    subgroup = df[(df["gender"] == 0) & (df["smoker_status"] == 0)]
    subgroup_X = subgroup.drop(columns=["time_to_event", "event"])

    mean_life_expectancy = []
    for i in subgroup.index:
        fn = surv_funcs[i]
        mean_survival = np.trapz(fn.y, fn.x)
        mean_life_expectancy.append(mean_survival)

    return round(np.mean(mean_life_expectancy), 2)

if __name__ == "__main__":
    df, X, y = load_data()

    print("Evaluating CoxPH...")
    cox_results = evaluate_cox(df, y)

    print("Evaluating RSF...")
    rsf_results = evaluate_rsf(X, y)

    print("Model Comparison:")
    print(cox_results)
    print(rsf_results)

    print("\n Bonus: Mean Life Expectancy (Non-Smoker Females):")
    print(f"{mean_life_expectancy_rsf(X, df)} months")
