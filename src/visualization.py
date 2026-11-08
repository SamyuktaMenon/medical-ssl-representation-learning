"""Visualization module for scientific figures, confusion matrices, and embedding projections."""

import os
import matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def plot_loss_curves(history: dict, title: str, save_path: str) -> None:
    """Plots training and validation loss curves over epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2, markersize=4)
    if "val_loss" in history and history["val_loss"]:
        plt.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2, markersize=4)
    plt.title(f"{title} - Loss Trajectory", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str,
    save_path: str,
    normalize: bool = True,
) -> None:
    """Plots and saves a high-resolution confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = np.asarray(cm)
    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
        fmt = ".2f"
        data_to_plot = cm_norm
    else:
        fmt = "d"
        data_to_plot = cm

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        data_to_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Proportion" if normalize else "Counts"},
    )
    plt.title(f"Confusion Matrix: {title}", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Ground Truth Class", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_label_efficiency_curves(
    df_results: pd.DataFrame,
    save_path: str,
    metric: str = "test_macro_f1",
) -> None:
    """Plots label efficiency comparison curve (metric vs labeled training fraction)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 6))

    method_palette = {
        "Supervised Scratch": "#d95f02",
        "ImageNet Transfer": "#7570b3",
        "SimCLR Linear Probe": "#1b9e77",
        "SimCLR Fine-Tuning": "#e7298a",
    }

    # Proportions as percentages for x-axis
    proportions = [1, 5, 10, 100]

    for method in df_results["method"].unique():
        sub_df = df_results[df_results["method"] == method].sort_values("proportion")
        x_vals = [p * 100 for p in sub_df["proportion"]]
        y_vals = sub_df[metric].values
        color = method_palette.get(method, None)
        plt.plot(
            x_vals,
            y_vals,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=method,
            color=color,
        )

    plt.xscale("log")
    plt.xticks(proportions, [f"{p}%" for p in proportions], fontsize=11)
    metric_label = metric.replace("test_", "").replace("_", " ").title()
    plt.xlabel("Proportion of Labeled Training Data (Log Scale)", fontsize=12, fontweight="bold")
    plt.ylabel(f"External Test Set {metric_label}", fontsize=12, fontweight="bold")
    plt.title(
        f"Label Efficiency Benchmark: Histopathological Image Classification\n({metric_label} vs Label Availability)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="Method", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison_bar(
    df_results: pd.DataFrame,
    save_path: str,
    proportions: list = [0.01, 0.05, 0.10, 1.00],
) -> None:
    """Plots grouped bar chart comparing methods across proportions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(proportions), figsize=(16, 5), sharey=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    method_palette = {
        "Supervised Scratch": "#d95f02",
        "ImageNet Transfer": "#7570b3",
        "SimCLR Linear Probe": "#1b9e77",
        "SimCLR Fine-Tuning": "#e7298a",
    }

    for idx, prop in enumerate(proportions):
        ax = axes[idx]
        sub_df = df_results[df_results["proportion"] == prop]
        bars = ax.bar(
            sub_df["method"],
            sub_df["test_macro_f1"],
            color=[method_palette.get(m, "#333333") for m in sub_df["method"]],
            alpha=0.85,
            edgecolor="black",
        )
        ax.set_title(f"{int(prop*100)}% Labeled Data", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(sub_df)))
        ax.set_xticklabels(
            [m.replace(" ", "\n") for m in sub_df["method"]],
            rotation=0,
            fontsize=9,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("External Test Macro-F1", fontsize=11, fontweight="bold")
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.suptitle("Comparative Evaluation Across Label Scarcity Regimes", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tsne_embeddings(
    embeddings_dict: dict,
    labels: np.ndarray,
    class_names: list,
    save_path: str,
) -> None:
    """Computes and plots 2D t-SNE projections comparing representations across models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5.5), sharex=True, sharey=True)
    if num_models == 1:
        axes = [axes]

    # Palette with 9 distinct colors for the 9 PathMNIST tissue classes
    palette = sns.color_palette("tab10", n_colors=9)

    for idx, (model_name, emb) in enumerate(embeddings_dict.items()):
        ax = axes[idx]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        z_2d = tsne.fit_transform(emb)

        for c_idx, c_name in enumerate(class_names):
            mask = labels == c_idx
            ax.scatter(
                z_2d[mask, 0],
                z_2d[mask, 1],
                color=palette[c_idx],
                label=c_name if idx == num_models - 1 else None,
                alpha=0.6,
                s=16,
                edgecolors="none",
            )
        ax.set_title(f"Representation Space:\n{model_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
        if idx == 0:
            ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Place common legend outside
    fig.legend(
        title="Histopathology Tissue Class",
        loc="center left",
        bbox_to_anchor=(0.91, 0.5),
        fontsize=9,
        title_fontsize=10,
    )
    plt.suptitle(
        "Latent Embedding Geometry (External Test Cohort CRC-VAL-HE-7K)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

