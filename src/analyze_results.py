# src/analyze_results.py

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"D:/YASH/CODES/PROJECTS/Research Project")

# Configuration grid
INIT_SCALES = [0.5, 1.0, 2.0]
USE_BN_OPTIONS = [False, True]
SEEDS = [42, 123, 999]

def collect_final_metrics():

    results = []

    for scale in INIT_SCALES:
        for use_bn in USE_BN_OPTIONS:

            acc_list = []
            depth_list = []
            gsi_list = []

            for seed in SEEDS:

                file_path = PROJECT_ROOT / f"log_scale_{scale}_bn_{use_bn}_seed_{seed}.csv"

                df = pd.read_csv(file_path)

                # Final epoch metrics
                final_row = df.iloc[-1]

                acc_list.append(final_row["Val_Acc"])
                depth_list.append(final_row["Depth_Ratio"])
                gsi_list.append(final_row["GSI"])

            results.append({
                "Scale": scale,
                "BatchNorm": use_bn,
                "Acc_mean": np.mean(acc_list),
                "Acc_std": np.std(acc_list),
                "Depth_mean": np.mean(depth_list),
                "Depth_std": np.std(depth_list),
                "GSI_mean": np.mean(gsi_list),
                "GSI_std": np.std(gsi_list)
            })

    return pd.DataFrame(results)


def main():

    df = collect_final_metrics()

    print("\n===== Aggregated Results =====\n")
    print(df)

    output_path = PROJECT_ROOT / "aggregated_results.csv"
    df.to_csv(output_path, index=False)

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()