"""Master experimental runner for self-supervised representation learning benchmark.

Orchestrates:
1. SimCLR pretraining on 89,996 unlabeled PathMNIST training images.
2. Downstream benchmarking across 4 methods:
   - Supervised ResNet-18 trained from scratch
   - ImageNet-pretrained ResNet-18 transfer learning
   - SimCLR-pretrained ResNet-18 linear probing
   - SimCLR-pretrained ResNet-18 full fine-tuning
3. Labeled training data proportions: 1%, 5%, 10%, 100%.
4. Zero-leakage protocol: Validation set used for model selection; Test set evaluated once.
5. Exporting metrics CSV, confusion matrices, loss curves, and t-SNE embedding visualizations.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed, get_device, load_config, setup_logger
from src.datasets import get_downstream_loaders
from src.models import SupervisedResNet18, ResNet18Encoder
from src.train_supervised import train_supervised_model
from src.pretrain_simclr import pretrain_simclr
from src.finetune import train_linear_probe, train_full_finetune
from src.evaluate import evaluate_checkpoint, extract_embeddings
from src.visualization import (
    plot_loss_curves,
    plot_confusion_matrix,
    plot_label_efficiency_curves,
    plot_model_comparison_bar,
    plot_tsne_embeddings,
)


def run_single_experiment(
    method: str,
    proportion: float,
    seed: int,
    config: dict,
    encoder_ckpt_path: str,
    device: torch.device,
    logger,
) -> dict:
    """Executes a single downstream experimental condition."""
    set_seed(seed)
    exp_name = f"{method.replace(' ', '_').lower()}_prop{int(proportion*100):03d}_seed{seed}"
    logger.info(f"\n{'='*70}\n[EXPERIMENT] {method} | Labels: {proportion*100:.1f}% | Seed: {seed}\n{'='*70}")

    # 1. Load exact data subsets (strictly train split, identical indices for same seed & proportion)
    train_loader, val_loader, test_loader, class_counts = get_downstream_loaders(
        config=config,
        label_fraction=proportion,
        seed=seed,
    )
    total_train_samples = sum(class_counts.values())
    logger.info(f"Loaded {total_train_samples} training samples across 9 classes: {class_counts}")

    class_names = [config["dataset"]["classes"][i] for i in range(config["dataset"]["num_classes"])]
    checkpoints_dir = os.path.join(PROJECT_ROOT, config.get("project", {}).get("checkpoints_dir", "checkpoints"))
    plots_dir = os.path.join(PROJECT_ROOT, "results", "plots")
    metrics_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    best_ckpt_path = os.path.join(checkpoints_dir, f"{exp_name}_best.pth")

    # Time tracking with MPS synchronization
    if device.type == "mps":
        torch.mps.synchronize()
    start_time = time.time()

    # 2. Train model according to experimental condition
    if method == "Supervised Scratch":
        model = SupervisedResNet18(num_classes=9, pretrained=False).to(device)
        model, history, best_val_metrics = train_supervised_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            experiment_name=exp_name,
            is_imagenet=False,
            logger=logger,
        )

    elif method == "ImageNet Transfer":
        model = SupervisedResNet18(num_classes=9, pretrained=True).to(device)
        model, history, best_val_metrics = train_supervised_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            experiment_name=exp_name,
            is_imagenet=True,
            logger=logger,
        )

    elif method == "SimCLR Linear Probe":
        model, history, best_val_metrics = train_linear_probe(
            encoder_checkpoint_path=encoder_ckpt_path,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            experiment_name=exp_name,
            logger=logger,
        )

    elif method == "SimCLR Fine-Tuning":
        model, history, best_val_metrics = train_full_finetune(
            encoder_checkpoint_path=encoder_ckpt_path,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            experiment_name=exp_name,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if device.type == "mps":
        torch.mps.synchronize()
    training_time = time.time() - start_time

    # 3. Save loss curves
    loss_curve_path = os.path.join(plots_dir, f"loss_curve_{exp_name}.png")
    plot_loss_curves(history, title=f"{method} ({int(proportion*100)}% labels)", save_path=loss_curve_path)

    # 4. Final Evaluation strictly on the untouched external test set using the best model
    logger.info(f"Evaluating best checkpoint on official external test set (CRC-VAL-HE-7K, N={len(test_loader.dataset)})...")
    test_metrics, y_true, y_pred, y_prob = evaluate_checkpoint(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    # 5. Save confusion matrix
    cm_path = os.path.join(plots_dir, f"confusion_matrix_{exp_name}.png")
    plot_confusion_matrix(
        cm=test_metrics["confusion_matrix"],
        class_names=class_names,
        title=f"{method} ({int(proportion*100)}% Labels)",
        save_path=cm_path,
    )

    auc_val = test_metrics.get("macro_roc_auc")
    auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"

    logger.info(
        f"[TEST RESULT] {exp_name} | "
        f"Test Acc: {test_metrics['accuracy']:.4f} | "
        f"Bal Acc: {test_metrics['balanced_accuracy']:.4f} | "
        f"Macro F1: {test_metrics['macro_f1']:.4f} | "
        f"ROC-AUC: {auc_str} | "
        f"Time: {training_time:.1f}s"
    )

    # Return structured record
    record = {
        "method": method,
        "proportion": proportion,
        "seed": seed,
        "n_samples": total_train_samples,
        "training_time_sec": round(training_time, 2),
        "val_best_macro_f1": round(best_val_metrics.get("macro_f1", 0.0), 4),
        "val_best_accuracy": round(best_val_metrics.get("accuracy", 0.0), 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_balanced_accuracy": round(test_metrics["balanced_accuracy"], 4),
        "test_macro_precision": round(test_metrics["macro_precision"], 4),
        "test_macro_recall": round(test_metrics["macro_recall"], 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_macro_roc_auc": round(test_metrics["macro_roc_auc"], 4) if test_metrics.get("macro_roc_auc") is not None else None,
        "checkpoint_path": best_ckpt_path,
        "per_class_f1": test_metrics.get("per_class_f1", []),
        "per_class_precision": test_metrics.get("per_class_precision", []),
        "per_class_recall": test_metrics.get("per_class_recall", []),
        "class_counts": class_counts,
    }

    # Save individual experiment JSON
    json_path = os.path.join(metrics_dir, f"metrics_{exp_name}.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    return record, model


def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner for Medical SSL Benchmark")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip SimCLR pretraining if checkpoint exists")
    parser.add_argument("--pretrain-epochs", type=int, default=15, help="Number of SimCLR pretraining epochs")
    parser.add_argument("--downstream-epochs", type=int, default=15, help="Downstream epochs per run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds for evaluation")
    parser.add_argument("--proportions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 1.00], help="Label proportions to benchmark")
    parser.add_argument("--methods", nargs="+", type=str, default=["Supervised Scratch", "ImageNet Transfer", "SimCLR Linear Probe", "SimCLR Fine-Tuning"], help="Methods to benchmark")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = load_config(config_path)

    # Set downstream epochs
    config["downstream"]["epochs"] = args.downstream_epochs
    config["simclr_pretrain"]["epochs"] = args.pretrain_epochs

    logger = setup_logger("Medical-SSL-Benchmark")
    device = get_device(config.get("hardware", {}).get("device", "auto"))
    logger.info(f"Initialized Experiment Suite on Device: {device}")

    checkpoints_dir = os.path.join(PROJECT_ROOT, config.get("project", {}).get("checkpoints_dir", "checkpoints"))
    metrics_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    plots_dir = os.path.join(PROJECT_ROOT, "results", "plots")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    encoder_best_path = os.path.join(checkpoints_dir, "simclr_encoder_best.pth")

    # =========================================================================
    # Step 1: Self-Supervised Contrastive Pretraining (SimCLR)
    # =========================================================================
    if not os.path.exists(encoder_best_path) or not args.skip_pretrain:
        logger.info("\n" + "#"*70 + "\n# STEP 1: SimCLR Self-Supervised Pretraining (89,996 Unlabeled Images)\n" + "#"*70)
        set_seed(config.get("project", {}).get("seed", 42))
        _, pretrain_history = pretrain_simclr(
            config=config,
            max_epochs=args.pretrain_epochs,
            logger=logger,
        )
        # Plot pretraining loss
        pretrain_plot_path = os.path.join(plots_dir, "simclr_pretrain_loss.png")
        plot_loss_curves({"train_loss": pretrain_history["loss"]}, title="SimCLR Pretraining (NT-Xent)", save_path=pretrain_plot_path)
    else:
        logger.info(f"SimCLR pretrained encoder checkpoint found at {encoder_best_path}. Skipping pretraining.")

    # =========================================================================
    # Step 2: Controlled Downstream Evaluations Across Methods and Proportions
    # =========================================================================
    methods = args.methods
    proportions = args.proportions

    all_records = []
    # If existing results CSV exists, load it to preserve previous results when extending
    csv_path = os.path.join(metrics_dir, "results_summary.csv")
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        all_records = existing_df.to_dict("records")
    trained_models_for_tsne = {}

    total_runs = len(methods) * len(proportions) * len(args.seeds)
    run_idx = 1

    for seed in args.seeds:
        for prop in proportions:
            for method in methods:
                logger.info(f"\n[Progress: {run_idx}/{total_runs}] Running {method} with {prop*100:.0f}% labels (seed {seed})")
                record, model = run_single_experiment(
                    method=method,
                    proportion=prop,
                    seed=seed,
                    config=config,
                    encoder_ckpt_path=encoder_best_path,
                    device=device,
                    logger=logger,
                )
                all_records.append(record)
                run_idx += 1

                # Incremental CSV saving after every run to guarantee persistence
                df_interim = pd.DataFrame(all_records)
                if not df_interim.empty and "method" in df_interim.columns:
                    df_interim = df_interim.drop_duplicates(subset=["method", "proportion", "seed"], keep="last")
                df_interim.to_csv(csv_path, index=False)

                # Save 100% models for t-SNE embedding comparison
                if prop == 1.00 and seed == 42:
                    trained_models_for_tsne[method] = model

    # =========================================================================
    # Step 3: Export Consolidated Results Table
    # =========================================================================
    df_results = pd.DataFrame(all_records)
    if not df_results.empty and "method" in df_results.columns:
        df_results = df_results.drop_duplicates(subset=["method", "proportion", "seed"], keep="last")
    csv_path = os.path.join(metrics_dir, "results_summary.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\nSaved consolidated metrics to {csv_path}")

    # =========================================================================
    # Step 4: Generate Publication-Quality Visualizations
    # =========================================================================
    logger.info("\nGenerating comparative evaluation plots...")
    # Seed 42 benchmark plots
    df_seed42 = df_results[df_results["seed"] == 42]

    plot_label_efficiency_curves(
        df_seed42,
        save_path=os.path.join(plots_dir, "label_efficiency_macro_f1.png"),
        metric="test_macro_f1",
    )
    plot_label_efficiency_curves(
        df_seed42,
        save_path=os.path.join(plots_dir, "label_efficiency_accuracy.png"),
        metric="test_accuracy",
    )
    plot_model_comparison_bar(
        df_seed42,
        save_path=os.path.join(plots_dir, "model_comparison_bar.png"),
        proportions=proportions,
    )

    # t-SNE Embeddings comparison
    if trained_models_for_tsne:
        logger.info("Extracting feature representations from external test set for t-SNE...")
        _, _, test_loader, _ = get_downstream_loaders(config, label_fraction=1.00, seed=42)
        embeddings_dict = {}
        class_names = [config["dataset"]["classes"][i] for i in range(config["dataset"]["num_classes"])]

        for m_name, m_obj in trained_models_for_tsne.items():
            emb, test_y = extract_embeddings(m_obj, test_loader, device=device, max_samples=1500)
            embeddings_dict[m_name] = emb

        tsne_path = os.path.join(plots_dir, "tsne_representations.png")
        plot_tsne_embeddings(embeddings_dict, test_y, class_names, save_path=tsne_path)
        logger.info(f"Saved t-SNE embedding projections to {tsne_path}")

    logger.info("\n" + "="*70 + "\nALL PHASE 4 CONTROLLED EXPERIMENTS SUCCESSFULLY COMPLETED!\n" + "="*70)


if __name__ == "__main__":
    main()

