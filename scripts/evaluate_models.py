#!/usr/bin/env python3
"""Evaluate trained checkpoints on the official external test cohort (CRC-VAL-HE-7K)."""

import os
import sys
import argparse
import glob
import torch
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config, get_device
from src.datasets import get_downstream_loaders
from src.models import ResNet18Encoder, SupervisedResNet18, LinearProbeClassifier, FineTuneClassifier
from src.evaluate import evaluate_checkpoint


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Loads appropriate model architecture and weights from checkpoint path."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    filename = os.path.basename(checkpoint_path)

    if "supervised_scratch" in filename or "imagenet_transfer" in filename:
        model = SupervisedResNet18(num_classes=9, pretrained=False).to(device)
    elif "simclr_linear_probe" in filename:
        encoder = ResNet18Encoder(pretrained=False)
        model = LinearProbeClassifier(encoder=encoder, num_classes=9).to(device)
    elif "simclr_fine-tuning" in filename or "finetune" in filename:
        encoder = ResNet18Encoder(pretrained=False)
        model = FineTuneClassifier(encoder=encoder, num_classes=9).to(device)
    else:
        raise ValueError(f"Could not infer architecture from checkpoint filename: {filename}")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on PathMNIST external test set")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific .pth checkpoint")
    parser.add_argument("--all", action="store_true", help="Evaluate all 16 downstream checkpoints in checkpoints/")
    args = parser.parse_args()

    config = load_config(os.path.join(PROJECT_ROOT, args.config))
    device = get_device("auto")
    print(f"Using device: {device}")

    # Load test set (official external test cohort CRC-VAL-HE-7K)
    _, _, test_loader, _ = get_downstream_loaders(config, label_fraction=1.00, seed=42)
    class_names = [config["dataset"]["classes"][i] for i in range(config["dataset"]["num_classes"])]
    print(f"Loaded test cohort: {len(test_loader.dataset)} samples across {len(class_names)} classes.\n")

    checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    if args.checkpoint:
        ckpts_to_eval = [args.checkpoint]
    elif args.all:
        pattern = os.path.join(checkpoints_dir, "*_prop*_seed*_best.pth")
        ckpts_to_eval = sorted(glob.glob(pattern))
    else:
        # Default: evaluate representative checkpoints (e.g., 1% label regime & 100% label regime)
        ckpts_to_eval = sorted(glob.glob(os.path.join(checkpoints_dir, "*_prop001_seed42_best.pth")))
        print("Note: Evaluating 1% label regime models by default. Use --all to evaluate all 16 models, or --checkpoint <path> for a specific model.\n")

    if not ckpts_to_eval:
        print("No matching checkpoints found to evaluate.")
        return

    records = []
    for ckpt_path in ckpts_to_eval:
        ckpt_name = os.path.basename(ckpt_path).replace("_best.pth", "")
        print(f"Evaluating {ckpt_name}...")
        model = load_model_from_checkpoint(ckpt_path, device)
        metrics, _, _, _ = evaluate_checkpoint(model, test_loader, device, class_names=class_names)

        records.append({
            "Model Condition": ckpt_name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Balanced Acc": f"{metrics['balanced_accuracy']:.4f}",
            "Macro F1": f"{metrics['macro_f1']:.4f}",
            "ROC-AUC": f"{metrics['macro_roc_auc']:.4f}" if metrics.get("macro_roc_auc") is not None else "N/A",
        })

    df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("EXTERNAL TEST SET (CRC-VAL-HE-7K, N=7,180) EVALUATION RESULTS:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

