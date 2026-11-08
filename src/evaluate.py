"""Evaluation module for comprehensive testing, metric computation, and representation extraction."""

import os
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import compute_metrics
from src.utils import get_device


@torch.no_grad()
def evaluate_checkpoint(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[list] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates a trained model on the official test set.

    Args:
        model: Trained PyTorch classifier.
        test_loader: DataLoader for the untouched official test set.
        device: Compute device.
        class_names: Optional list of class names.

    Returns:
        metrics: Dictionary of classification metrics.
        y_true: True labels array.
        y_pred: Predicted labels array.
        y_prob: Predicted probability array.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names=class_names)
    return metrics, y_true, y_pred, y_prob


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts penultimate feature embeddings h from the model's encoder.

    Args:
        model: Classifier model with extract_features() method or encoder attribute.
        loader: DataLoader.
        device: Compute device.
        max_samples: Maximum number of samples to extract for t-SNE / visualization.

    Returns:
        embeddings: Array of shape (N, 512).
        labels: Array of shape (N,).
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    total = 0

    for images, targets in loader:
        if total >= max_samples:
            break
        images = images.to(device)

        if hasattr(model, "extract_features"):
            feats = model.extract_features(images)
        elif hasattr(model, "encoder"):
            feats = model.encoder(images)
        elif callable(model):
            feats = model(images)
        else:
            raise AttributeError("Model has neither extract_features nor encoder attribute.")

        all_embeddings.append(feats.cpu().numpy())
        all_labels.append(targets.numpy())
        total += images.size(0)

    embeddings = np.concatenate(all_embeddings, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    return embeddings, labels

