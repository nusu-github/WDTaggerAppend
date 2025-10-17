"""Metrics for WD Tagger training and evaluation.

This module provides metric computation functions compatible with Hugging Face Trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchmetrics.classification import AUROC, F1Score, MatthewsCorrCoef, Precision, Recall

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import EvalPrediction


# Default threshold from JAX-CV implementation
DEFAULT_THRESHOLD = 0.4

__all__ = [
    "DEFAULT_THRESHOLD",
    "create_compute_metrics_fn",
]


def create_compute_metrics_fn(
    num_labels: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Create a compute_metrics function for use with Hugging Face Trainer.

    Matches the JAX-CV implementation:
    - F1: macro-averaged (average across labels)
    - MCC: macro-averaged (average across labels)
    - Precision: micro-averaged (global TP/FP across all samples and labels)
    - Recall: micro-averaged (global TP/FN across all samples and labels)
    - AUROC: macro-averaged (average across labels)

    Args:
        num_labels: Total number of labels in the model
        threshold: Probability threshold for binary classification (default: 0.4)

    Returns:
        Function that computes metrics from EvalPrediction

    Example:
        >>> compute_metrics = create_compute_metrics_fn(num_labels=9667)
        >>> trainer = Trainer(
        ...     model=model,
        ...     compute_metrics=compute_metrics,
        ...     ...
        ... )
    """
    # Initialize metrics matching JAX-CV configuration
    f1_metric = F1Score(
        task="multilabel",
        num_labels=num_labels,
        average="macro",  # JAX-CV: macro averaging
        threshold=threshold,
    )
    mcc_metric = MatthewsCorrCoef(
        task="multilabel",
        num_labels=num_labels,
        threshold=threshold,  # JAX-CV: macro averaging (torchmetrics default for multilabel)
    )
    precision_metric = Precision(
        task="multilabel",
        num_labels=num_labels,
        average="micro",  # JAX-CV: micro averaging (changed from macro)
        threshold=threshold,
    )
    recall_metric = Recall(
        task="multilabel",
        num_labels=num_labels,
        average="micro",  # JAX-CV: micro averaging (changed from macro)
        threshold=threshold,
    )
    auroc_metric = AUROC(
        task="multilabel",
        num_labels=num_labels,
        average="macro",
    )

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        """Compute metrics from model predictions.

        Follows JAX-CV behavior:
        1. Sigmoid is applied to logits (from_logits=True in JAX-CV)
        2. Probabilities are compared against threshold for binary predictions
        3. Labels are compared against threshold (for soft label support)
        4. Metrics are computed on the binarized predictions and labels

        Note: Currently labels are hard (0 or 1), so label > threshold
        is equivalent to label == 1. This preserves compatibility if
        soft labels are introduced in the future.

        Args:
            eval_pred: Predictions and labels from evaluation

        Returns:
            Dictionary of metric names to values
        """
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

        # Convert to tensors
        logits_tensor = torch.from_numpy(logits).to(torch.float32)
        labels_float = torch.from_numpy(labels).to(torch.float32)
        labels_int = labels_float.to(torch.int64)
        logits_tensor = torch.sigmoid(logits_tensor)

        # Compute metrics on binarized predictions and labels
        f1_score = f1_metric(logits_tensor, labels_int)
        mcc_score = mcc_metric(logits_tensor, labels_int)
        precision_score = precision_metric(logits_tensor, labels_int)
        recall_score = recall_metric(logits_tensor, labels_int)
        auroc_value = auroc_metric(logits_tensor, labels_int)

        # Reset torchmetrics state to avoid leakage across evaluation cycles
        auroc_metric.reset()
        f1_metric.reset()
        mcc_metric.reset()
        precision_metric.reset()
        recall_metric.reset()

        return {
            "auroc": float(auroc_value.item()),
            "f1": float(f1_score.item()),
            "mcc": float(mcc_score.item()),
            "precision": float(precision_score.item()),
            "recall": float(recall_score.item()),
        }

    return compute_metrics
