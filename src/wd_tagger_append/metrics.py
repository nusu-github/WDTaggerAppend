"""Metrics for WD Tagger training and evaluation.

This module provides metric computation functions compatible with Hugging Face Trainer.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torchmetrics.classification import AUROC, F1Score, MatthewsCorrCoef, Precision, Recall
from transformers import EvalPrediction

# Default threshold from JAX-CV implementation
DEFAULT_THRESHOLD = 0.4

__all__ = [
    "DEFAULT_THRESHOLD",
    "MetricFactory",
    "MetricSettings",
    "create_compute_metrics_fn",
]


@dataclass(frozen=True)
class MetricSettings:
    """Configuration for metric computation."""

    num_labels: int
    threshold: float = DEFAULT_THRESHOLD


class MetricFactory:
    """Build compute_metrics functions compatible with Hugging Face Trainer."""

    def __init__(self, settings: MetricSettings) -> None:
        self._settings = settings

    def build(self) -> Callable[[EvalPrediction], dict[str, float]]:
        """Return a callable that computes evaluation metrics."""
        f1_metric = F1Score(
            task="multilabel",
            num_labels=self._settings.num_labels,
            average="micro",
            threshold=self._settings.threshold,
        )
        mcc_metric = MatthewsCorrCoef(
            task="multilabel",
            num_labels=self._settings.num_labels,
            threshold=self._settings.threshold,
        )
        precision_metric = Precision(
            task="multilabel",
            num_labels=self._settings.num_labels,
            average="micro",
            threshold=self._settings.threshold,
        )
        recall_metric = Recall(
            task="multilabel",
            num_labels=self._settings.num_labels,
            average="micro",
            threshold=self._settings.threshold,
        )
        auroc_metric = AUROC(
            task="multilabel",
            num_labels=self._settings.num_labels,
            average="weighted",
        )

        def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
            logits_tensor = torch.from_numpy(eval_pred.predictions).to(torch.float32)
            labels_float = torch.from_numpy(eval_pred.label_ids).to(torch.float32)
            labels_int = labels_float.to(torch.int64)
            logits_tensor = torch.sigmoid(logits_tensor)

            f1_score = f1_metric(logits_tensor, labels_int)
            mcc_score = mcc_metric(logits_tensor, labels_int)
            precision_score = precision_metric(logits_tensor, labels_int)
            recall_score = recall_metric(logits_tensor, labels_int)
            auroc_value = auroc_metric(logits_tensor, labels_int)

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


def create_compute_metrics_fn(
    num_labels: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Compatibility wrapper exposing the previous functional API."""
    return MetricFactory(MetricSettings(num_labels=num_labels, threshold=threshold)).build()
