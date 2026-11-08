"""Statistical aggregation and report generator for Medical SSL experiments."""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config


def generate_summary_tables(csv_path: str, config_path: str, output_dir: str):
    """Aggregates experiment runs across seeds and generates Markdown and LaTeX summary tables."""
    if not os.path.exists(csv_path):
        print(f"Metrics CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    config = load_config(config_path)
    class_names = [config["dataset"]["classes"][i] for i in range(config["dataset"]["num_classes"])]

    # 1. Main aggregation across seeds
    group_cols = ["method", "proportion"]
    metrics_to_agg = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_roc_auc",
        "training_time_sec",
    ]

    agg_dict = {}
    for m in metrics_to_agg:
        if m in df.columns:
            agg_dict[f"{m}_mean"] = (m, "mean")
            agg_dict[f"{m}_std"] = (m, "std")

    summary_df = df.groupby(group_cols).agg(**agg_dict).reset_index()
    for col in summary_df.columns:
        if col.endswith("_std"):
            summary_df[col] = summary_df[col].fillna(0.0)

    # 2. Generate formatted Markdown Table
    md_lines = [
        "# Consolidated Experimental Results: Self-Supervised vs. Supervised & Transfer Learning",
        "",
        "| Method | Labeled Data (%) | N Samples | Test Acc (Mean ± Std) | Bal Acc (Mean ± Std) | Macro F1 (Mean ± Std) | Macro ROC-AUC | Train Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in summary_df.iterrows():
        method = row["method"]
        prop_pct = f"{int(row['proportion'] * 100)}%"
        n_samples = int(row["proportion"] * 89996)

        acc_str = f"{row['test_accuracy_mean']:.4f} ± {row['test_accuracy_std']:.4f}"
        bal_acc_str = f"{row['test_balanced_accuracy_mean']:.4f} ± {row['test_balanced_accuracy_std']:.4f}"
        f1_str = f"{row['test_macro_f1_mean']:.4f} ± {row['test_macro_f1_std']:.4f}"
        auc_str = f"{row['test_macro_roc_auc_mean']:.4f}" if pd.notnull(row.get("test_macro_roc_auc_mean")) else "N/A"
        time_str = f"{row['training_time_sec_mean']:.1f}s"

        md_lines.append(
            f"| **{method}** | {prop_pct} | {n_samples:,} | {acc_str} | {bal_acc_str} | **{f1_str}** | {auc_str} | {time_str} |"
        )

    md_output_path = os.path.join(output_dir, "summary_table.md")
    with open(md_output_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Generated Markdown summary table: {md_output_path}")

    # 3. Generate LaTeX Table
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Downstream classification performance on external test set CRC-VAL-HE-7K under varying label availability regimes.}",
        r"\label{tab:downstream_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Labels (\%)} & \textbf{N Samples} & \textbf{Test Accuracy} & \textbf{Balanced Acc} & \textbf{Macro F1} & \textbf{ROC-AUC} \\",
        r"\midrule",
    ]

    for _, row in summary_df.iterrows():
        method = row["method"]
        prop_pct = f"{int(row['proportion'] * 100)}\\%"
        n_samples = f"{int(row['proportion'] * 89996):,}"
        acc_str = f"{row['test_accuracy_mean']:.4f} $\\pm$ {row['test_accuracy_std']:.4f}"
        bal_acc_str = f"{row['test_balanced_accuracy_mean']:.4f} $\\pm$ {row['test_balanced_accuracy_std']:.4f}"
        f1_str = f"\\textbf{{{row['test_macro_f1_mean']:.4f} $\\pm$ {row['test_macro_f1_std']:.4f}}}"
        auc_str = f"{row['test_macro_roc_auc_mean']:.4f}" if pd.notnull(row.get("test_macro_roc_auc_mean")) else "N/A"

        latex_lines.append(
            f"{method} & {prop_pct} & {n_samples} & {acc_str} & {bal_acc_str} & {f1_str} & {auc_str} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex_output_path = os.path.join(output_dir, "summary_table.tex")
    with open(latex_output_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"Generated LaTeX summary table: {latex_output_path}")

    return summary_df


if __name__ == "__main__":
    csv_file = os.path.join(PROJECT_ROOT, "results", "metrics", "results_summary.csv")
    cfg_file = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    out_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    generate_summary_tables(csv_file, cfg_file, out_dir)

