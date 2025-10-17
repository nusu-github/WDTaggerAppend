"""Dataset utility functions for WD Tagger training.

This module provides utilities for label encoding, dataset transformation,
and integration with training pipelines.
"""

from __future__ import annotations

import numbers
from collections import Counter
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

from datasets import Dataset, Image

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence
    from pathlib import Path


VALID_TAG_CATEGORIES: tuple[str, ...] = ("rating", "general", "character")

RATING_CODE_TO_NAME = {
    "g": "general",
    "s": "sensitive",
    "q": "questionable",
    "e": "explicit",
}


# WD Tagger category codes (matching original selected_tags.csv format)
CATEGORY_MAP = {
    "rating": 9,
    "general": 0,
    "character": 4,
}


def _validate_categories(categories: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize and validate category selections."""
    if categories is None:
        return ("general", "character")

    invalid = [category for category in categories if category not in VALID_TAG_CATEGORIES]
    if invalid:
        invalid_str = ", ".join(sorted(invalid))
        msg = f"Unsupported tag categories: {invalid_str}"
        raise ValueError(msg)

    normalized: list[str] = []
    seen: set[str] = set()
    for category in categories:
        if category in seen:
            continue
        normalized.append(category)
        seen.add(category)

    return tuple(normalized)


def _normalize_rating_value(value: Any) -> str | None:
    """Return a normalized rating tag string or ``None`` if unavailable."""
    if value is None:
        return None

    # ClassLabel values may arrive as ints; we cannot recover the string name here.
    if isinstance(value, numbers.Integral):
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_rating_value(item)
            if normalized is not None:
                return normalized
        return None

    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if not candidate:
        return None

    return RATING_CODE_TO_NAME.get(candidate, candidate)


def load_allowed_tags(path: Path) -> set[str]:
    """Load a newline-delimited allow list of tags."""
    allowed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            tag = raw_line.strip()
            if not tag or tag.startswith("#"):
                continue
            allowed.add(tag)
    return allowed


def count_tag_frequencies(
    dataset: Dataset,
    categories: Sequence[str] | None = None,
) -> Counter[str]:
    """Count tag occurrences for the selected categories.

    Optimized implementation using Dataset.map with batched=True for performance.
    """
    active_categories = _validate_categories(categories)
    frequencies: Counter[str] = Counter()

    def extract_tags_batch(examples: dict[str, Any]) -> dict[str, list[list[str]]]:
        """Extract tags from a batch of examples."""
        batch_size = len(examples.get("tags", []))
        all_tags = []

        for i in range(batch_size):
            tags_dict = examples["tags"][i] if "tags" in examples else {}
            example_tags = []

            if "general" in active_categories:
                example_tags.extend(tag for tag in tags_dict.get("general", []) if tag)

            if "character" in active_categories:
                example_tags.extend(tag for tag in tags_dict.get("character", []) if tag)

            if "rating" in active_categories:
                rating_sources = (
                    examples.get("rating", [None])[i] if "rating" in examples else None,
                    tags_dict.get("rating"),
                )
                for source in rating_sources:
                    rating_tag = _normalize_rating_value(source)
                    if rating_tag is not None:
                        example_tags.append(rating_tag)
                        break

            all_tags.append(example_tags)

        return {"extracted_tags": all_tags}

    # Use map with batched=True for performance
    dataset = dataset.cast_column("image", Image(decode=False))
    processed = dataset.map(extract_tags_batch, batched=True)

    # Aggregate frequencies
    for example in processed:
        tags_list = example["extracted_tags"]  # type: ignore[index]
        for tag in tags_list:
            frequencies[tag] += 1

    return frequencies


def determine_tag_categories(
    dataset: Dataset,
    categories: Sequence[str] | None = None,
) -> dict[str, str]:
    """Determine which category each tag belongs to in the dataset.

    For tags appearing in multiple categories, the first occurrence wins
    (priority: rating > general > character).

    Optimized implementation using Dataset.map with batched=True for performance.

    Args:
        dataset: Dataset containing tags in nested structure.
        categories: Iterable of tag categories to consider.

    Returns:
        Dictionary mapping tag names to category names ('rating', 'general', 'character').
    """
    active_categories = _validate_categories(categories)
    tag_to_category: dict[str, str] = {}

    def extract_categorized_tags_batch(
        examples: dict[str, Any],
    ) -> dict[str, list[dict[str, list[str]]]]:
        """Extract tags by category from a batch of examples."""
        batch_size = len(examples.get("tags", []))
        categorized_batch = []

        for i in range(batch_size):
            tags_dict = examples["tags"][i] if "tags" in examples else {}
            categorized = {"rating": [], "general": [], "character": []}

            # Priority order: rating first (most specific)
            if "rating" in active_categories:
                rating_sources = (
                    examples.get("rating", [None])[i] if "rating" in examples else None,
                    tags_dict.get("rating") if isinstance(tags_dict, dict) else None,
                )
                for source in rating_sources:
                    rating_tag = _normalize_rating_value(source)
                    if rating_tag is not None:
                        categorized["rating"].append(rating_tag)
                        break

            # Then general tags
            if "general" in active_categories and isinstance(tags_dict, dict):
                categorized["general"].extend(tag for tag in tags_dict.get("general", []) if tag)

            # Finally character tags
            if "character" in active_categories and isinstance(tags_dict, dict):
                categorized["character"].extend(
                    tag for tag in tags_dict.get("character", []) if tag
                )

            categorized_batch.append(categorized)

        return {"categorized_tags": categorized_batch}

    # Use map with batched=True for performance
    dataset = dataset.cast_column("image", Image(decode=False))
    processed = dataset.map(extract_categorized_tags_batch, batched=True)

    # Aggregate with priority (first occurrence wins)
    for example in processed:
        categorized = example["categorized_tags"]  # type: ignore[index]

        # Priority: rating > general > character
        for tag in categorized.get("rating", []):
            if tag not in tag_to_category:
                tag_to_category[tag] = "rating"

        for tag in categorized.get("general", []):
            if tag not in tag_to_category:
                tag_to_category[tag] = "general"

        for tag in categorized.get("character", []):
            if tag not in tag_to_category:
                tag_to_category[tag] = "character"

    return tag_to_category


def create_label_mapping(
    dataset: Dataset,
    categories: Sequence[str] | None = None,
    *,
    min_count: int = 1,
    allowed_tags: Collection[str] | None = None,
) -> tuple[dict[str, int], Counter[str]]:
    """Create a mapping from tag strings to label indices.

    Args:
        dataset: Dataset containing tags in nested structure.
        categories: Iterable of tag categories to consider.
        min_count: Minimum number of occurrences required to keep a tag.
        allowed_tags: Optional allow list restricting tags to the provided set.

    Returns:
        Tuple of (label_mapping, tag_frequencies).
    """
    if min_count < 1:
        msg = "min_count must be >= 1"
        raise ValueError(msg)

    active_categories = _validate_categories(categories)
    frequencies = count_tag_frequencies(dataset, active_categories)

    filtered_tags = [
        tag
        for tag, count in frequencies.items()
        if count >= min_count and (allowed_tags is None or tag in allowed_tags)
    ]

    sorted_tags = sorted(filtered_tags)
    mapping = {tag: idx for idx, tag in enumerate(sorted_tags)}
    return mapping, frequencies


def save_labels_as_csv(
    label_list: list[str],
    tag_categories: dict[str, list[str]],
    output_path: Path,
) -> None:
    """Save labels in WD Tagger v3 compatible CSV format (selected_tags.csv).

    Args:
        label_list: Ordered list of all label names.
        tag_categories: Dictionary mapping category names to lists of tags in that category.
                       Expected keys: 'rating', 'general', 'character'
        output_path: Path to save the CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build category lookup
    tag_to_category: dict[str, int] = {}
    for category_name, tags in tag_categories.items():
        category_code = CATEGORY_MAP.get(category_name)
        if category_code is None:
            continue
        for tag in tags:
            tag_to_category[tag] = category_code

    # Write CSV
    df = pd.DataFrame(
        {
            "tag_id": list(range(len(label_list))),
            "name": label_list,
            "category": [tag_to_category.get(tag, CATEGORY_MAP["general"]) for tag in label_list],
            "count": [0] * len(label_list),  # Placeholder; not computed from dataset
        },
    )

    df.to_csv(output_path, index=False)


def encode_multi_labels(
    tags_dict: Mapping[str, Any],
    label_mapping: dict[str, int],
    categories: Sequence[str],
    *,
    rating_value: Any | None = None,
) -> torch.Tensor:
    """Encode tags as a multi-hot vector respecting category filters."""
    num_classes = len(label_mapping)
    labels = torch.zeros(num_classes, dtype=torch.float32)

    def add_tag(tag: str | None) -> None:
        if tag is None:
            return
        if tag in label_mapping:
            labels[label_mapping[tag]] = 1.0

    if "general" in categories:
        for tag in tags_dict.get("general", []):
            add_tag(tag)

    if "character" in categories:
        for tag in tags_dict.get("character", []):
            add_tag(tag)

    if "rating" in categories:
        rating_candidates: list[Any] = []
        if rating_value is not None:
            rating_candidates.append(rating_value)

        rating_field = tags_dict.get("rating")
        if rating_field is not None:
            rating_candidates.append(rating_field)

        for candidate in rating_candidates:
            normalized = _normalize_rating_value(candidate)
            if normalized is not None:
                add_tag(normalized)
                break

    return labels


def create_transform_function(
    transform: Callable,
    label_mapping: dict[str, int],
    categories: Sequence[str],
) -> Callable:
    """Create a transform function for use with Dataset.set_transform.

    Args:
        transform: Transform function to apply to images (from augmentation.py).
        label_mapping: Dictionary mapping tag strings to indices.
        categories: Sequence of tag categories to encode.

    Returns:
        Transform function compatible with Dataset.set_transform.
    """
    active_categories = _validate_categories(categories)

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
        ratings = examples.get("rating")
        rating_values = ratings if isinstance(ratings, list) else None

        labels = []
        for index, tags in enumerate(examples["tags"]):
            rating_value = rating_values[index] if rating_values is not None else None
            encoded = encode_multi_labels(
                tags,
                label_mapping,
                active_categories,
                rating_value=rating_value,
            )
            labels.append(encoded)
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
        "tag_counts": {"general": 0, "character": 0, "rating": 0},
        "rating_distribution": {},
    }

    # Collect statistics
    for example in dataset:
        # Count tags per category
        tags_dict = example["tags"]  # type: ignore[index]
        for category in stats["tag_counts"]:
            if category in ("general", "character"):
                stats["tag_counts"][category] += len(tags_dict.get(category, []))

        if "rating" in stats["tag_counts"]:
            rating_tag = example.get("rating")  # type: ignore[index]
            normalized = _normalize_rating_value(rating_tag)
            if normalized is not None:
                stats["tag_counts"]["rating"] += 1

        # Count ratings
        rating = example["rating"]  # type: ignore[index]
        stats["rating_distribution"][rating] = stats["rating_distribution"].get(rating, 0) + 1

    return stats
