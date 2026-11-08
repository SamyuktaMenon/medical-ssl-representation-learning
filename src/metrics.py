"""Evaluation metrics for multi-class classification and representation assessment."""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted class labels, shape (N,).
        y_prob: Predicted class probabilities, shape (N, num_classes).
        class_names: List of class names for reporting.

    Returns:
        Dictionary of computed scalar metrics and structured summaries.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    num_classes = len(class_names) if class_names is not None else int(max(np.max(y_true), np.max(y_pred)) + 1)
    class_labels = list(range(num_classes))

    # Per-class metrics
    per_class_prec = precision_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0).tolist()
    per_class_rec = recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0).tolist()
    per_class_f1 = f1_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=class_labels).tolist()

    # Macro ROC-AUC calculation
    roc_auc = None
    if y_prob is not None:
        try:
            # Multi-class One-vs-Rest ROC-AUC
            roc_auc = float(
                roc_auc_score(
                    y_true,
                    y_prob,
                    labels=class_labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except Exception:
            roc_auc = None

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "macro_roc_auc": roc_auc,
        "per_class_precision": per_class_prec,
        "per_class_recall": per_class_rec,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }

    if class_names is not None:
        try:
            report = classification_report(
                y_true,
                y_pred,
                labels=class_labels,
                target_names=class_names,
                zero_division=0,
                output_dict=True,
            )
            metrics["classification_report"] = report
        except Exception:
            pass

    return metrics

