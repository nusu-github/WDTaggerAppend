"""Dataset utility functions for WD Tagger training.

This module provides utilities for label encoding, dataset transformation,
and integration with training pipelines.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from datasets import Dataset


def create_label_mapping(dataset: Dataset) -> dict[str, int]:
    """Create a mapping from tag strings to label indices.

    Args:
        dataset: Dataset containing tags in nested structure.

    Returns:
        Dictionary mapping tag strings to integer indices.
    """
    all_tags = set()

    # Collect all unique tags from all categories
    for example in dataset:
        tags_dict = example["tags"]
        for category in ["general", "character", "copyright", "artist", "meta"]:
            all_tags.update(tags_dict[category])

    # Create sorted mapping for reproducibility
    sorted_tags = sorted(all_tags)
    return {tag: idx for idx, tag in enumerate(sorted_tags)}


def save_label_mapping(label_mapping: dict[str, int], output_path: Path) -> None:
    """Save label mapping to a JSON file.

    Args:
        label_mapping: Dictionary mapping tags to indices.
        output_path: Path to save the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)


def load_label_mapping(mapping_path: Path) -> dict[str, int]:
    """Load label mapping from a JSON file.

    Args:
        mapping_path: Path to the JSON file containing the mapping.

    Returns:
        Dictionary mapping tag strings to integer indices.

    Raises:
        FileNotFoundError: If the mapping file doesn't exist.
    """
    if not mapping_path.exists():
        msg = f"Label mapping file not found: {mapping_path}"
        raise FileNotFoundError(msg)

    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_multi_labels(
    tags_dict: dict[str, list[str]],
    label_mapping: dict[str, int],
) -> torch.Tensor:
    """Encode tags as multi-hot vector.

    Args:
        tags_dict: Dictionary with tag categories as keys and tag lists as values.
        label_mapping: Dictionary mapping tag strings to indices.

    Returns:
        Multi-hot encoded tensor of shape (num_classes,).
    """
    num_classes = len(label_mapping)
    labels = torch.zeros(num_classes, dtype=torch.float32)

    # Collect all tags from all categories
    all_tags = []
    for category in ["general", "character", "copyright", "artist", "meta"]:
        all_tags.extend(tags_dict.get(category, []))

    # Set corresponding indices to 1
    for tag in all_tags:
        if tag in label_mapping:
            labels[label_mapping[tag]] = 1.0

    return labels


def create_transform_function(
    transform: Callable,
    label_mapping: dict[str, int],
) -> Callable:
    """Create a transform function for use with Dataset.set_transform.

    Args:
        transform: Transform function to apply to images (from augmentation.py).
        label_mapping: Dictionary mapping tag strings to indices.

    Returns:
        Transform function compatible with Dataset.set_transform.
    """

    def transform_function(examples: dict) -> dict:
        """Apply transforms and encode labels.

        Args:
            examples: Batch of examples from the dataset.

        Returns:
            Transformed examples with pixel_values and labels.
        """
        # Apply image transforms
        images = [transform(img.convert("RGB")) for img in examples["image"]]
        examples["pixel_values"] = torch.stack(images)

        # Encode multi-labels
        labels = [encode_multi_labels(tags, label_mapping) for tags in examples["tags"]]
        examples["labels"] = torch.stack(labels)

        # Remove unnecessary fields to save memory
        del examples["image"]

        return examples

    return transform_function


def get_dataset_statistics(dataset: Dataset) -> dict:
    """Compute statistics about the dataset.

    Args:
        dataset: Dataset to analyze.

    Returns:
        Dictionary containing dataset statistics.
    """
    stats = {
        "num_examples": len(dataset),
        "tag_counts": {"general": 0, "character": 0, "copyright": 0, "artist": 0, "meta": 0},
        "rating_distribution": {},
    }

    # Collect statistics
    for example in dataset:
        # Count tags per category
        tags_dict = example["tags"]
        for category in stats["tag_counts"]:
            stats["tag_counts"][category] += len(tags_dict[category])

        # Count ratings
        rating = example["rating"]
        stats["rating_distribution"][rating] = stats["rating_distribution"].get(rating, 0) + 1

    return stats
