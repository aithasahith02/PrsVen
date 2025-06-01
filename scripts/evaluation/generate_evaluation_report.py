from evaluate_models import get_model_evaluation_summary
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# AUC Curve
def plot_auc_curves(times, cox_aucs, rsf_aucs, output_dir):
    plt.figure()
    plt.plot(times, cox_aucs, label="CoxPH AUC", marker='o')
    plt.plot(times, rsf_aucs, label="RSF AUC", marker='x')
    plt.title("Time-dependent AUC")
    plt.xlabel("Time (months)")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "auc_comparison.png"))
    plt.close()

# Brier Scores
def plot_brier_scores(times, cox_bs, rsf_bs, output_dir):
    plt.figure()
    plt.plot(times, cox_bs, label="CoxPH Brier Score", marker='o')
    plt.plot(times, rsf_bs, label="RSF Brier Score", marker='x')
    plt.title("Brier Scores Over Time")
    plt.xlabel("Time (months)")
    plt.ylabel("Brier Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "brier_comparison.png"))
    plt.close()

# C-Index & IBS bar plot
def plot_metric_bars(cox_metrics, rsf_metrics, output_dir):
    metrics = ["C-Index", "IBS"]
    values = [
        [cox_metrics["C-Index"], cox_metrics["IBS"]],
        [rsf_metrics["C-Index"], rsf_metrics["IBS"]]
    ]
    df = pd.DataFrame(values, columns=metrics, index=["CoxPH", "RSF"])
    df.plot(kind="bar")
    plt.title("Model Comparison - C-Index and IBS")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_bar.png"))
    plt.close()

# Individual survival curves
def plot_survival_curves(surv_funcs, output_dir, num_curves=5):
    plt.figure()
    for i, fn in enumerate(surv_funcs[:num_curves]):
        plt.step(fn.x, fn.y, where="post", label=f"Patient {i}")
    plt.title(f"Survival Curves (First {num_curves} Patients)")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_survival_curves.png"))
    plt.close()

# Report generator
def generate_report(data, output_dir="report_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    model_metrics = pd.DataFrame(data["models"])
    print("\nModel Performance:\n")
    print(model_metrics)

    model_metrics.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("Model Performance Summary\n\n")
        f.write(model_metrics.to_string(index=False))
        if "bonus" in data:
            f.write("\n\nBonus Info:\n")
            for k, v in data["bonus"].items():
                f.write(f"{k}: {v}\n")

    # Combined metric bar plot (seaborn)
    melted = model_metrics.melt(id_vars=["Model"], var_name="Metric", value_name="Value")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="Metric", y="Value", hue="Model")
    plt.title("Model Metric Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_comparison.png"))
    plt.close()

    # Additional plots
    plot_auc_curves(data["times"], data["cox_auc"], data["rsf_auc"], output_dir)
    plot_brier_scores(data["times"], data["cox_brier"], data["rsf_brier"], output_dir)
    plot_metric_bars(data["models"][0], data["models"][1], output_dir)
    plot_survival_curves(data["rsf_surv_funcs"], output_dir)

    print(f"\nReport generated at: {output_dir}")

# Entrypoint
if __name__ == "__main__":
    summary = get_model_evaluation_summary()
    generate_report(summary)
