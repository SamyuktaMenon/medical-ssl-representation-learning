"""Downstream fine-tuning and linear probing module for pretrained SimCLR encoders."""

import os
import copy
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models import ResNet18Encoder, LinearProbeClassifier, FineTuneClassifier
from src.train_supervised import train_one_epoch, evaluate_model
from src.utils import setup_logger, save_checkpoint, get_device


def train_linear_probe(
    encoder_checkpoint_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
    experiment_name: str,
    logger=None,
    max_epochs: Optional[int] = None,
) -> Tuple[LinearProbeClassifier, Dict[str, list], Dict[str, Any]]:
    """Trains a linear classification head on top of a frozen pretrained encoder."""
    if logger is None:
        logger = setup_logger(experiment_name)

    logger.info(f"Setting up Linear Probing from checkpoint: {encoder_checkpoint_path}")

    # Load pretrained encoder
    encoder = ResNet18Encoder(pretrained=False)
    state_dict = torch.load(encoder_checkpoint_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state_dict)

    # Initialize Linear Probe Classifier (freezes encoder parameters internally)
    num_classes = config["dataset"].get("num_classes", 9)
    model = LinearProbeClassifier(encoder=encoder, num_classes=num_classes).to(device)

    # Verify that all encoder parameters have requires_grad == False
    for name, param in model.encoder.named_parameters():
        assert not param.requires_grad, f"Encoder parameter {name} must be frozen in linear probing!"
    assert model.fc.weight.requires_grad, "Linear probe classifier head must be trainable!"

    epochs = max_epochs if max_epochs is not None else config["downstream"].get("epochs", 25)
    patience = config["downstream"].get("early_stopping_patience", 7)
    lr = config["downstream"].get("linear_probe", {}).get("learning_rate", 2e-3)
    weight_decay = config["downstream"].get("linear_probe", {}).get("weight_decay", 1e-5)
    checkpoints_dir = config.get("project", {}).get("checkpoints_dir", "checkpoints")
    if not os.path.isabs(checkpoints_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoints_dir = os.path.normpath(os.path.join(project_root, checkpoints_dir))
    os.makedirs(checkpoints_dir, exist_ok=True)

    class_names = [config["dataset"]["classes"][i] for i in range(num_classes)]
    criterion = nn.CrossEntropyLoss()

    # Optimize ONLY the linear classification head
    optimizer = optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
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

    checkpoint_path = os.path.join(checkpoints_dir, f"{experiment_name}_best.pth")
    logger.info(f"Starting Linear Probing for {epochs} epochs on {device}")

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
            f"Probe Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

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
                logger.info(f"Linear probing early stopping at epoch {epoch}. Best Val F1: {best_val_f1:.4f}")
                break

    model.load_state_dict(best_model_weights)
    logger.info(f"Linear Probing completed. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    return model, history, best_val_metrics


def train_full_finetune(
    encoder_checkpoint_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
    experiment_name: str,
    logger=None,
    max_epochs: Optional[int] = None,
) -> Tuple[FineTuneClassifier, Dict[str, list], Dict[str, Any]]:
    """Fully fine-tunes the pretrained encoder and classification head with differential learning rates."""
    if logger is None:
        logger = setup_logger(experiment_name)

    logger.info(f"Setting up Full Fine-Tuning from checkpoint: {encoder_checkpoint_path}")

    # Load pretrained encoder
    encoder = ResNet18Encoder(pretrained=False)
    state_dict = torch.load(encoder_checkpoint_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state_dict)

    # Initialize FineTuneClassifier (all parameters trainable)
    num_classes = config["dataset"].get("num_classes", 9)
    model = FineTuneClassifier(encoder=encoder, num_classes=num_classes).to(device)

    epochs = max_epochs if max_epochs is not None else config["downstream"].get("epochs", 25)
    patience = config["downstream"].get("early_stopping_patience", 7)
    checkpoints_dir = config.get("project", {}).get("checkpoints_dir", "checkpoints")
    if not os.path.isabs(checkpoints_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoints_dir = os.path.normpath(os.path.join(project_root, checkpoints_dir))
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Differential learning rates: lower for backbone, higher for classification head
    backbone_lr = config["downstream"].get("finetune", {}).get("backbone_lr", 1e-4)
    head_lr = config["downstream"].get("finetune", {}).get("head_lr", 1e-3)
    weight_decay = config["downstream"].get("finetune", {}).get("weight_decay", 1e-4)

    class_names = [config["dataset"]["classes"][i] for i in range(num_classes)]
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": backbone_lr},
            {"params": model.fc.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
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

    checkpoint_path = os.path.join(checkpoints_dir, f"{experiment_name}_best.pth")
    logger.info(
        f"Starting Full Fine-Tuning for {epochs} epochs on {device} "
        f"(Backbone LR: {backbone_lr}, Head LR: {head_lr})"
    )

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
            f"FineTune Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"Backbone LR: {current_lr:.6f}"
        )

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
                logger.info(f"Fine-tuning early stopping at epoch {epoch}. Best Val F1: {best_val_f1:.4f}")
                break

    model.load_state_dict(best_model_weights)
    logger.info(f"Full Fine-Tuning completed. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    return model, history, best_val_metrics
