"""Metrics for WD Tagger training and evaluation.

This module provides metric computation functions compatible with Hugging Face Trainer.
Based on the original JAX-CV implementation which uses:
- F1 Score (macro-averaged, threshold=0.4)
- Matthews Correlation Coefficient (macro-averaged, threshold=0.4)
- Binary Cross Entropy Loss (with sigmoid)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import BCEWithLogitsLoss
from torchmetrics import F1Score, MatthewsCorrCoef, Precision, Recall

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import EvalPrediction


# Default threshold from JAX-CV implementation
DEFAULT_THRESHOLD = 0.4


def create_compute_metrics_fn(
    num_labels: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Create a compute_metrics function for use with Hugging Face Trainer.

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
    # Initialize metrics with proper configuration
    f1_metric = F1Score(
        task="multilabel",
        num_labels=num_labels,
        average="macro",
        threshold=threshold,
    )
    mcc_metric = MatthewsCorrCoef(
        task="multilabel",
        num_labels=num_labels,
        threshold=threshold,
    )
    precision_metric = Precision(
        task="multilabel",
        num_labels=num_labels,
        average="macro",
        threshold=threshold,
    )
    recall_metric = Recall(
        task="multilabel",
        num_labels=num_labels,
        average="macro",
        threshold=threshold,
    )
    bce_metric = BCEWithLogitsLoss()

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        """Compute metrics from model predictions.

        Args:
            eval_pred: Predictions and labels from evaluation

        Returns:
            Dictionary of metric names to values
        """
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

        # Convert to tensors
        logits_tensor = torch.from_numpy(logits)
        labels_tensor = torch.from_numpy(labels)

        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits_tensor)

        # Compute metrics
        f1_score = f1_metric(probs, labels_tensor.int())
        mcc_score = mcc_metric(probs, labels_tensor.int())
        precision_score = precision_metric(probs, labels_tensor.int())
        recall_score = recall_metric(probs, labels_tensor.int())
        bce_score = bce_metric(logits_tensor, labels_tensor.float())

        return {
            "f1": float(f1_score.item()),
            "mcc": float(mcc_score.item()),
            "precision": float(precision_score.item()),
            "recall": float(recall_score.item()),
            "bce": float(bce_score.item()),
        }

    return compute_metrics
