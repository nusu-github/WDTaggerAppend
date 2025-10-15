"""Utilities for constructing frozen teacher models used in consistency training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForImageClassification

from wd_tagger_append.model_export import configure_model_for_remote

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from wd_tagger_append.labels import ModelLabels


def build_teacher_model(
    *,
    model_repo_id: str,
    extended_labels: ModelLabels,
    original_weight: torch.Tensor,
    original_bias: torch.Tensor,
    new_weight_rows: torch.Tensor,
    new_bias_values: torch.Tensor,
) -> PreTrainedModel:
    """Create a frozen teacher model that mirrors the student's head expansion."""
    label2id, id2label = extended_labels.to_label_mappings()
    teacher = AutoModelForImageClassification.from_pretrained(
        model_repo_id,
        num_labels=extended_labels.num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    configure_model_for_remote(teacher)

    with torch.no_grad():
        combined_weight = torch.cat([original_weight, new_weight_rows], dim=0)
        combined_bias = torch.cat([original_bias, new_bias_values], dim=0)
        teacher.timm_model.head.weight.data = combined_weight.to(
            teacher.timm_model.head.weight.dtype,
        )
        teacher.timm_model.head.bias.data = combined_bias.to(teacher.timm_model.head.bias.dtype)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher
