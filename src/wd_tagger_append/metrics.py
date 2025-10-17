"""Metrics for WD Tagger training and evaluation.

This module provides metric computation functions compatible with Hugging Face Trainer
along with helpers for ranking metrics and continual learning diagnostics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torchmetrics import F1Score, MatthewsCorrCoef, Precision, Recall
from torchmetrics.classification import AUROC

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from transformers import EvalPrediction


# Default threshold from JAX-CV implementation
DEFAULT_THRESHOLD = 0.4

__all__ = [
    "DEFAULT_THRESHOLD",
    "compute_average_forgetting",
    "compute_forgetting_scores",
    "compute_propensity_scores",
    "create_compute_metrics_fn",
]


def compute_propensity_scores(
    label_frequencies: Sequence[int],
    dataset_size: int,
    *,
    a: float = 0.55,
    b: float = 1.5,
) -> torch.Tensor:
    """Compute propensity scores for propensity-scored ranking metrics.

    Implements the propensity formulation from Jain et al. (2016):
    ``p_j = 1 + C * (n_j + b)^{-a}`` where ``C = (log(n) - 1) * (b + 1)^{a}``.

    Args:
        label_frequencies: Positive example counts per label (aligned with model head).
        dataset_size: Total number of training examples used to estimate frequencies.
        a: Propensity hyper-parameter ``a``.
        b: Propensity hyper-parameter ``b``.

    Returns:
        Tensor of propensity scores ordered to match ``label_frequencies``.

    Raises:
        ValueError: If ``dataset_size`` is not positive.
    """
    if dataset_size <= 0:
        msg = "dataset_size must be a positive integer when computing propensity scores."
        raise ValueError(msg)

    counts = torch.tensor(label_frequencies, dtype=torch.float32)
    constant = (math.log(dataset_size) - 1.0) * (b + 1.0) ** a
    # Clamp constant to zero when log(dataset_size) < 1 to avoid negative scaling for tiny corpora.
    constant = max(constant, 0.0)
    propensities = 1.0 + constant * torch.pow(counts + b, -a)
    # Propensities must remain positive to avoid unstable inverse weights.
    return propensities.clamp(min=1e-12)


def compute_forgetting_scores(metric_history: Sequence[Sequence[float]]) -> list[float]:
    """Compute per-task forgetting scores for continual learning setups.

    Each entry in ``metric_history`` should contain the evaluation metric for a task measured
    after successive training phases (chronological order). The forgetting score for a task is
    defined as ``max(history[:-1]) - history[-1]`` (Lopez-Paz & Ranzato, 2017). Tasks with a
    single measurement are assumed to have zero forgetting.

    Args:
        metric_history: Iterable where each inner iterable holds the metric trajectory for a
            single task.

    Returns:
        List of forgetting scores ordered to match ``metric_history``.

    Raises:
        ValueError: If any task history is empty.
    """
    scores: list[float] = []
    for task_history in metric_history:
        history = list(task_history)
        if not history:
            msg = "Metric history entries must contain at least one measurement."
            raise ValueError(msg)
        if len(history) == 1:
            scores.append(0.0)
            continue
        peak = max(history[:-1])
        scores.append(max(peak - history[-1], 0.0))
    return scores


def compute_average_forgetting(metric_history: Sequence[Sequence[float]]) -> float:
    """Return the average forgetting score across tasks."""
    scores = compute_forgetting_scores(metric_history)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def create_compute_metrics_fn(
    num_labels: int,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: Sequence[int] | None = None,
    label_frequencies: Sequence[int] | None = None,
    dataset_size: int | None = None,
    propensity_a: float = 0.55,
    propensity_b: float = 1.5,
    enable_propensity: bool = False,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Create a compute_metrics function for use with Hugging Face Trainer.

    Matches the JAX-CV implementation:
    - F1: macro-averaged (average across labels)
    - MCC: macro-averaged (average across labels)
    - Precision: micro-averaged (global TP/FP across all samples and labels)
    - Recall: micro-averaged (global TP/FN across all samples and labels)

    Optional extensions:
    - Ranking metrics at configured cut-offs (Precision@K, Recall@K, nDCG@K)
    - Propensity-scored nDCG@K for XMLC diagnostics
    - Macro-averaged AUROC for incremental / continual learning tracking

    Args:
        num_labels: Total number of labels in the model
        threshold: Probability threshold for binary classification (default: 0.4)
        top_k: Optional sequence of cut-offs for ranking metrics.
        label_frequencies: Positive counts per label (used for propensity scoring).
        dataset_size: Dataset size used to estimate propensities.
        propensity_a: Propensity hyper-parameter ``a``.
        propensity_b: Propensity hyper-parameter ``b``.
        enable_propensity: Enable propensity-scored ranking metrics.

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
    # Deduplicate and validate top-k configuration once for efficiency
    top_k_values: tuple[int, ...] = ()
    if top_k:
        unique_sorted = sorted({int(value) for value in top_k if value > 0})
        if not unique_sorted:
            msg = "top_k values must be positive integers."
            raise ValueError(msg)
        if unique_sorted[-1] > num_labels:
            msg = "top_k values cannot exceed the number of classifier outputs."
            raise ValueError(msg)
        top_k_values = tuple(unique_sorted)

    if enable_propensity:
        if label_frequencies is None:
            msg = "label_frequencies must be provided when enable_propensity is True."
            raise ValueError(msg)
        if len(label_frequencies) != num_labels:
            msg = "label_frequencies must match num_labels when computing propensities."
            raise ValueError(msg)
        if dataset_size is None:
            msg = "dataset_size must be supplied when enable_propensity is True."
            raise ValueError(msg)

    propensity_tensor: torch.Tensor | None = None
    if enable_propensity and label_frequencies is not None and dataset_size is not None:
        propensity_tensor = compute_propensity_scores(
            label_frequencies,
            dataset_size,
            a=propensity_a,
            b=propensity_b,
        )

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
    auroc_metric = AUROC(task="multilabel", num_labels=num_labels, average="macro")

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
        probs_tensor = torch.sigmoid(logits_tensor)

        # Compute metrics on binarized predictions and labels
        f1_score = f1_metric(logits_tensor, labels_int)
        mcc_score = mcc_metric(logits_tensor, labels_int)
        precision_score = precision_metric(logits_tensor, labels_int)
        recall_score = recall_metric(logits_tensor, labels_int)
        auroc_value = auroc_metric(probs_tensor, labels_int)

        metrics: dict[str, float] = {
            "auroc": float(auroc_value.item()),
            "f1": float(f1_score.item()),
            "mcc": float(mcc_score.item()),
            "precision": float(precision_score.item()),
            "recall": float(recall_score.item()),
        }

        # Reset torchmetrics state to avoid leakage across evaluation cycles
        auroc_metric.reset()
        f1_metric.reset()
        mcc_metric.reset()
        precision_metric.reset()
        recall_metric.reset()

        if top_k_values:
            max_k = top_k_values[-1]
            _, topk_indices = probs_tensor.topk(max_k, dim=1)
            relevance = torch.take_along_dim(labels_float, topk_indices, dim=1)
            label_counts = labels_float.sum(dim=1)
            label_counts_long = labels_int.sum(dim=1)

            positions = torch.arange(1, max_k + 1, dtype=probs_tensor.dtype)
            discounts = 1.0 / torch.log2(positions + 1.0)
            discount_cumsum = torch.cumsum(discounts, dim=0)

            if propensity_tensor is not None:
                inv_prop = torch.reciprocal(propensity_tensor)
                inv_prop_topk = inv_prop[topk_indices]
                inv_prop_relevance = torch.where(
                    labels_float > 0,
                    inv_prop.unsqueeze(0),
                    torch.zeros_like(labels_float),
                )
                top_relevant_weights, _ = inv_prop_relevance.topk(max_k, dim=1)
            else:
                inv_prop_topk = None
                top_relevant_weights = None

            for k in top_k_values:
                rel_k = relevance[:, :k]
                hits = rel_k.sum(dim=1)
                precision_at_k = (hits / float(k)).mean()

                denominator = label_counts.clamp_min(1.0)
                recall_at_k = (hits / denominator).masked_fill(label_counts == 0, 0.0).mean()

                metrics[f"precision_at_{k}"] = float(precision_at_k.item())
                metrics[f"recall_at_{k}"] = float(recall_at_k.item())

                dcg = (rel_k * discounts[:k]).sum(dim=1)
                ideal_counts = torch.clamp(label_counts_long, max=k)
                ideal_dcg = torch.zeros_like(label_counts)
                ideal_mask = ideal_counts > 0
                if ideal_mask.any():
                    ideal_indices = ideal_counts[ideal_mask] - 1
                    ideal_dcg[ideal_mask] = discount_cumsum[ideal_indices]
                ndcg = torch.where(ideal_dcg > 0, dcg / ideal_dcg, torch.zeros_like(dcg))
                metrics[f"ndcg_at_{k}"] = float(ndcg.mean().item())

                if inv_prop_topk is not None and top_relevant_weights is not None:
                    ps_dcg = (rel_k * inv_prop_topk[:, :k] * discounts[:k]).sum(dim=1)
                    ideal_weights = top_relevant_weights[:, :k]
                    ps_ideal = (ideal_weights * discounts[:k]).sum(dim=1)
                    psndcg = torch.where(
                        ps_ideal > 0,
                        ps_dcg / ps_ideal,
                        torch.zeros_like(ps_dcg),
                    )
                    metrics[f"psndcg_at_{k}"] = float(psndcg.mean().item())

        return metrics

    return compute_metrics
