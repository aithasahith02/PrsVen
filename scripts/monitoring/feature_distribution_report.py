import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
import os
import glob
import datetime

REPORT_OUTPUT_DIR = "feature_distribution_reports" # Saving the reports here
DATA_LAKE_BASE_DIR = "./data-lake" # Local data lake path
TRAINING_DATA_PATH_PATTERN = os.path.join(DATA_LAKE_BASE_DIR, "training", "*.parquet")
INFERENCE_DATA_PATH_PATTERN = os.path.join(DATA_LAKE_BASE_DIR, "inference", "*.parquet")

def load_and_prepare_data() -> pd.DataFrame:
    print("Loading training data...")
    training_files = glob.glob(TRAINING_DATA_PATH_PATTERN)
    if not training_files:
        print(f"Warning: No training Parquet files found at {TRAINING_DATA_PATH_PATTERN}. Using simulated data.")
        df_train = pd.DataFrame({
            'gender': ['Male', 'Female'] * 50,
            'smoker_status': ['No', 'Yes'] * 50,
            'age': np.random.normal(55, 10, 100),
            **{f'feature_{i+1}': np.random.rand(100) for i in range(75)}
        })
    else:
        df_train = pd.read_parquet(training_files[0])
        print(f"Loaded training data from {training_files[0]}, shape: {df_train.shape}")

    df_train['source_type'] = 'training'

    print("Loading inference data...")
    inference_files = glob.glob(INFERENCE_DATA_PATH_PATTERN)
    if not inference_files:
        print(f"Warning: No inference Parquet files found at {INFERENCE_DATA_PATH_PATTERN}. Using simulated data.")
        df_inference = pd.DataFrame({
            'gender': ['Female', 'Male', 'Other'] * 40,
            'smoker_status': ['Yes', 'No'] * 60,
            'age': np.random.normal(60, 12, 120),
            **{f'feature_{i+1}': np.random.rand(120) * 1.1 for i in range(75)}
        })
    else:
        df_inference = pd.concat([pd.read_parquet(f) for f in inference_files], ignore_index=True)
        print(f"Loaded {len(inference_files)} inference files, total inference samples: {len(df_inference)}")

    df_inference['source_type'] = 'inference'

    cols_to_drop_from_inference = ['request_id', 'api_request_timestamp']
    for col_drop in cols_to_drop_from_inference:
        if col_drop in df_inference.columns:
            df_inference = df_inference.drop(columns=[col_drop])

    df_combined = pd.concat([df_train, df_inference], ignore_index=True)
    print(f"Combined data shape for reporting: {df_combined.shape}")
    return df_combined


def generate_numerical_feature_report(df_combined: pd.DataFrame, col: str, report_writer):
    report_writer.write(f"\n### Numerical Feature: `{col}`\n\n")
    
    summary_stats = df_combined.groupby('source_type')[col].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).rename(columns={'<lambda_0>': '25th_pct', '<lambda_1>': '75th_pct'})
    report_writer.write("Summary Statistics:\n```\n")
    report_writer.write(summary_stats.to_string() + "\n```\n\n")

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_combined, x=col, hue='source_type', kde=True, stat="density", common_norm=False, palette={'training':'skyblue', 'inference':'lightcoral'})
    plt.title(f'Distribution of {col} (Training vs. Inference)')
    plt.tight_layout()
    plot_filename = f"{col}_distribution_hist.png"
    plt.savefig(os.path.join(REPORT_OUTPUT_DIR, plot_filename))
    plt.close()
    report_writer.write(f"![Histogram for {col}](./{plot_filename})\n\n")

    training_data = df_combined[df_combined['source_type'] == 'training'][col].dropna()
    inference_data = df_combined[df_combined['source_type'] == 'inference'][col].dropna()
    if len(training_data) > 20 and len(inference_data) > 20: 
        ks_stat, p_value = ks_2samp(training_data, inference_data)
        report_writer.write(f"**Kolmogorov-Smirnov Test:** Stat = {ks_stat:.4f}, P-value = {p_value:.4g}\n")
        if p_value < 0.05:
            report_writer.write(f"  <span style='color:red;'>ALERT: Significant distribution difference detected (p < 0.05)</span>\n")
    else:
        report_writer.write("  (KS test skipped: insufficient data in one or both groups for a reliable test.)\n")
    report_writer.write("\n---\n")

def generate_categorical_feature_report(df_combined: pd.DataFrame, col: str, report_writer):
    report_writer.write(f"\n### Categorical Feature: `{col}`\n\n")

    if col not in df_combined.columns or df_combined[col].isnull().all() or df_combined.groupby('source_type')[col].nunique().sum() == 0 :
        report_writer.write(f"  (Skipped: Column `{col}` has no data or no variance across sources.)\n\n---\n")
        return

    try:
        perc_dist = pd.crosstab(df_combined[col], df_combined['source_type'], normalize='columns').mul(100).round(2)
        report_writer.write("Percentage Distribution:\n```\n")
        report_writer.write(perc_dist.to_string() + "\n```\n\n")

        plt.figure(figsize=(10, max(6, df_combined[col].nunique()*0.8) )) 
        plot_df = df_combined.groupby('source_type')[col].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
        sns.barplot(data=plot_df, x=col, y='percentage', hue='source_type', palette={'training':'skyblue', 'inference':'lightcoral'})
        plt.title(f'Distribution of {col} (Training vs. Inference)')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = f"{col}_distribution_bar.png"
        plt.savefig(os.path.join(REPORT_OUTPUT_DIR, plot_filename))
        plt.close()
        report_writer.write(f"![Bar Chart for {col}](./{plot_filename})\n\n")

        contingency_table = pd.crosstab(df_combined[col], df_combined['source_type'])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1 and contingency_table.sum().sum() > 5 and all(contingency_table.sum(axis=i) > 0 for i in [0,1]):
            chi2, p, _, _ = chi2_contingency(contingency_table)
            report_writer.write(f"**Chi-squared Test:** chi2 = {chi2:.2f}, p-value = {p:.4g}\n")
            if p < 0.05:
                report_writer.write(f"  <span style='color:red;'>ALERT: Significant difference detected (p < 0.05)</span>\n")
        else:
            report_writer.write(f"  (Chi-squared test skipped: contingency table not suitable for `{col}`.)\n")
    except Exception as e:
        report_writer.write(f"  (Error generating report for `{col}`: {e})\n")
    report_writer.write("\n---\n")


def create_feature_distribution_report():
    if not os.path.exists(REPORT_OUTPUT_DIR):
        os.makedirs(REPORT_OUTPUT_DIR)
    print(f"Starting feature distribution report generation. Output to: {REPORT_OUTPUT_DIR}")

    df_combined = load_and_prepare_data()

    if df_combined.empty:
        print("Combined dataframe is empty. Report generation aborted.")
        return
    
    categorical_cols = ['gender', 'smoker_status'] 
    numerical_cols = ['age'] + [f'feature_{i+1}' for i in range(75)] 

    report_filename = f"feature_comparison_report_{datetime.date.today()}.md"
    report_file_path = os.path.join(REPORT_OUTPUT_DIR, report_filename)

    with open(report_file_path, "w") as report_writer:
        report_writer.write(f"# Feature Distribution Comparison Report ({datetime.date.today()})\n\n")
        report_writer.write("Comparing training dataset features against production inference request features.\n\n")
        report_writer.write("## Data Summary\n```\n" + df_combined['source_type'].value_counts().to_string() + "\n```\n\n")
        
        report_writer.write("## Categorical Feature Distributions\n")
        report_writer.write(
            "_Note: For accurate categorical comparison, ensure training data provides 'gender' and 'smoker_status' "
            "as raw strings, similar to API inputs. If training data has them encoded, this comparison might be misleading "
            "or require pre-alignment._\n"
        )
        for col in categorical_cols:
            if col in df_combined.columns:
                generate_categorical_feature_report(df_combined, col, report_writer)
            else:
                print(f"Warning: Categorical column '{col}' not found. Skipping.")
                report_writer.write(f"\n*Warning: Categorical column `{col}` not found in combined data.*\n")
        
        report_writer.write("\n## Numerical Feature Distributions\n")
        for col in numerical_cols:
            if col in df_combined.columns:
                generate_numerical_feature_report(df_combined, col, report_writer)
            else:
                print(f"Warning: Numerical column '{col}' not found. Skipping.")
                report_writer.write(f"\n*Warning: Numerical column `{col}` not found in combined data.*\n")

    print(f"Report generation complete. Output saved to '{REPORT_OUTPUT_DIR}'. Main report: '{report_file_path}'")

if __name__ == "__main__":
    create_feature_distribution_report()