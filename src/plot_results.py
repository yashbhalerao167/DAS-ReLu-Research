# src/plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(r"D:/YASH/CODES/PROJECTS/Research Project")

def plot_metric(df, metric_mean, metric_std, ylabel, filename):

    plt.figure()

    for use_bn in [False, True]:

        subset = df[df["BatchNorm"] == use_bn]

        label = "BatchNorm" if use_bn else "No BatchNorm"

        plt.errorbar(
            subset["Scale"],
            subset[metric_mean],
            yerr=subset[metric_std],
            marker='o',
            capsize=5,
            label=label
        )

    plt.xlabel("Initialization Scale")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    save_path = PROJECT_ROOT / filename
    plt.savefig(save_path)
    plt.close()

    print("Saved:", save_path)


def main():

    df = pd.read_csv(PROJECT_ROOT / "aggregated_results.csv")

    plot_metric(
        df,
        "Acc_mean",
        "Acc_std",
        "Validation Accuracy",
        "figure_accuracy.png"
    )

    plot_metric(
        df,
        "Depth_mean",
        "Depth_std",
        "Depth Amplification Ratio",
        "figure_depth_ratio.png"
    )

    plot_metric(
        df,
        "GSI_mean",
        "GSI_std",
        "Gradient Stability Index (GSI)",
        "figure_gsi.png"
    )


if __name__ == "__main__":
    main()