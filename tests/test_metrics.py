"""Unit tests for metric helpers."""

from __future__ import annotations

import numpy as np
import pytest
from transformers import EvalPrediction

from wd_tagger_append.metrics import (
    compute_average_forgetting,
    compute_forgetting_scores,
    compute_propensity_scores,
    create_compute_metrics_fn,
)


def _to_logits(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits for stable test fixtures."""
    odds = np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1)
    return np.log(odds).astype(np.float32)


def test_ranking_metrics_basic() -> None:
    probabilities = np.array(
        [
            [0.90, 0.80, 0.10, 0.05, 0.02],
            [0.60, 0.40, 0.30, 0.20, 0.10],
        ],
        dtype=np.float32,
    )
    logits = _to_logits(probabilities)
    labels = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    compute_metrics = create_compute_metrics_fn(num_labels=5, threshold=0.5, top_k=(1, 3))
    metrics = compute_metrics(eval_pred)

    assert metrics["precision_at_1"] == pytest.approx(0.5, rel=1e-6)
    assert metrics["recall_at_1"] == pytest.approx(0.25, rel=1e-6)
    assert metrics["precision_at_3"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert metrics["recall_at_3"] == pytest.approx(1.0, rel=1e-6)

    discounts = 1.0 / np.log2(np.arange(1, 4) + 1)
    dcg_sample1 = discounts[0] + discounts[1]
    idcg_sample1 = dcg_sample1
    dcg_sample2 = discounts[1] + discounts[2]
    idcg_sample2 = discounts[0] + discounts[1]
    expected_ndcg = ((dcg_sample1 / idcg_sample1) + (dcg_sample2 / idcg_sample2)) / 2.0
    assert metrics["ndcg_at_3"] == pytest.approx(expected_ndcg, rel=1e-6)


def test_propensity_scored_ndcg_matches_reference() -> None:
    probabilities = np.array(
        [
            [0.92, 0.50, 0.30, 0.15, 0.05],
            [0.70, 0.45, 0.25, 0.20, 0.10],
        ],
        dtype=np.float32,
    )
    logits = _to_logits(probabilities)
    labels = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    label_frequencies = [500, 200, 5, 1, 1]
    dataset_size = 1000
    propensities = compute_propensity_scores(label_frequencies, dataset_size)
    inv_prop = 1.0 / propensities.numpy()

    topk_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
    relevance = labels[np.arange(labels.shape[0])[:, None], topk_indices]
    discounts = 1.0 / np.log2(np.arange(1, 4) + 1)
    ps_dcg = (relevance * inv_prop[topk_indices] * discounts).sum(axis=1)

    relevant_weights = inv_prop * labels
    sorted_weights = -np.sort(-relevant_weights, axis=1)[:, :3]
    ps_idcg = (sorted_weights * discounts).sum(axis=1)
    expected_psndcg = (ps_dcg / ps_idcg).mean()

    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    compute_metrics = create_compute_metrics_fn(
        num_labels=5,
        threshold=0.5,
        top_k=(3,),
        label_frequencies=label_frequencies,
        dataset_size=dataset_size,
        enable_propensity=True,
    )
    metrics = compute_metrics(eval_pred)

    assert metrics["psndcg_at_3"] == pytest.approx(expected_psndcg, rel=1e-5)


def test_forgetting_helpers() -> None:
    history = [
        [0.50, 0.62, 0.58],
        [0.71, 0.74, 0.69],
        [0.40],
    ]
    scores = compute_forgetting_scores(history)
    assert scores == pytest.approx([0.04, 0.05, 0.0])
    average = compute_average_forgetting(history)
    assert average == pytest.approx(sum(scores) / len(scores))
