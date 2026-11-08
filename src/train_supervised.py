"""Supervised training module for ResNet-18 baseline (Scratch) and ImageNet transfer learning."""

import os
import copy
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models import SupervisedResNet18
from src.metrics import compute_metrics
from src.utils import setup_logger, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[list] = None,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates model on validation or test DataLoader."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / max(total, 1)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names=class_names)
    return avg_loss, metrics, y_true, y_pred, y_prob


def train_supervised_model(
    model: SupervisedResNet18,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
    experiment_name: str,
    is_imagenet: bool = False,
    logger=None,
    max_epochs: Optional[int] = None,
) -> Tuple[SupervisedResNet18, Dict[str, list], Dict[str, Any]]:
    """Complete training loop for supervised ResNet-18 (Scratch or ImageNet)."""
    if logger is None:
        logger = setup_logger(experiment_name)

    epochs = max_epochs if max_epochs is not None else config["downstream"].get("epochs", 25)
    patience = config["downstream"].get("early_stopping_patience", 7)
    checkpoint_dir = config.get("project", {}).get("checkpoints_dir", "checkpoints")
    if not os.path.isabs(checkpoint_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_dir = os.path.normpath(os.path.join(project_root, checkpoint_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)

    class_names = [config["dataset"]["classes"][i] for i in range(config["dataset"]["num_classes"])]
    criterion = nn.CrossEntropyLoss()

    # Differential learning rates for ImageNet transfer learning
    if is_imagenet:
        backbone_lr = config["downstream"]["imagenet"].get("backbone_lr", 1e-4)
        head_lr = config["downstream"]["imagenet"].get("head_lr", 1e-3)
        weight_decay = config["downstream"]["imagenet"].get("weight_decay", 1e-4)
        optimizer = optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": backbone_lr},
                {"params": model.fc.parameters(), "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        lr = config["downstream"]["scratch"].get("learning_rate", 1e-3)
        weight_decay = config["downstream"]["scratch"].get("weight_decay", 1e-4)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_metrics = {}
    patience_counter = 0

    checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")

    logger.info(f"Starting training: {experiment_name} on {device} for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _, _ = evaluate_model(
            model, val_loader, criterion, device, class_names=class_names
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Early stopping and checkpointing on validation macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            patience_counter = 0

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs). "
                    f"Best epoch: {best_epoch} with Val F1: {best_val_f1:.4f}"
                )
                break

    # Load best weights
    model.load_state_dict(best_model_weights)
    logger.info(f"Training completed. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    return model, history, best_val_metrics

