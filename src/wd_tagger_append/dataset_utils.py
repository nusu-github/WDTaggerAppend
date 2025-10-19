"""Dataset utility functions for WD Tagger training.

This module provides utilities for label encoding, dataset transformation,
and integration with training pipelines using Pandas vectorized operations.
"""

from __future__ import annotations

import json
import numbers
from collections import Counter
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence
    from pathlib import Path

    from datasets import Dataset

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
    """Count tag occurrences using Pandas vectorized operations.

    Uses:
    - Pandas DataFrame.explode() for tag expansion
    - Pandas value_counts() for counting
    - Batch processing without Python-level iteration

    Args:
        dataset: Dataset containing tags in nested structure.
        categories: Tag categories to consider.

    Returns:
        Counter mapping tags to frequencies.
    """
    active_categories = _validate_categories(categories)

    # Convert to Pandas DataFrame (zero-copy from Arrow)
    # Don't decode images - we only need metadata
    df: pd.DataFrame = dataset.remove_columns("image").to_pandas()  # type: ignore

    all_tags: list[str] = []

    # Extract general tags using Pandas explode
    if "general" in active_categories:
        general_tags_series = df["tags"].apply(
            lambda x: x.get("general", []) if isinstance(x, dict) else [],
        )
        # Explode list of lists into flat series
        general_series: pd.Series = general_tags_series.explode()  # type: ignore
        all_tags.extend(general_series[general_series.notna()].tolist())

    # Extract character tags using Pandas explode
    if "character" in active_categories:
        character_tags_series = df["tags"].apply(
            lambda x: x.get("character", []) if isinstance(x, dict) else [],
        )
        character_series: pd.Series = character_tags_series.explode()  # type: ignore
        all_tags.extend(character_series[character_series.notna()].tolist())

    # Extract rating tags
    if "rating" in active_categories:
        rating_tags_series: pd.Series = df["rating"].apply(_normalize_rating_value)  # type: ignore
        all_tags.extend(rating_tags_series[rating_tags_series.notna()].tolist())

    # Count occurrences
    return Counter(all_tags)


def determine_tag_categories(
    dataset: Dataset,
    categories: Sequence[str] | None = None,
) -> dict[str, str]:
    """Determine which category each tag belongs to using Pandas operations.

    For tags appearing in multiple categories, the first occurrence wins
    (priority: rating > general > character).

    All tags are processed in a single DataFrame pass without repetition.

    Args:
        dataset: Dataset containing tags in nested structure.
        categories: Iterable of tag categories to consider.

    Returns:
        Dictionary mapping tag names to category names.
    """
    active_categories = _validate_categories(categories)

    # Convert to Pandas DataFrame (zero-copy from Arrow)
    df: pd.DataFrame = dataset.remove_columns("image").to_pandas()  # type: ignore

    tags_with_categories: list[tuple[str, str]] = []

    # Collect all tags with their categories - priority: rating > general > character
    if "rating" in active_categories:
        rating_tags_series: pd.Series = df["rating"].apply(_normalize_rating_value)  # type: ignore
        rating_exploded: pd.Series = (
            rating_tags_series[rating_tags_series.notna()].drop_duplicates()  # type: ignore
        )
        tags_with_categories.extend([(tag, "rating") for tag in rating_exploded])

    if "general" in active_categories:
        general_tags_series = df["tags"].apply(
            lambda x: x.get("general", []) if isinstance(x, dict) else [],
        )
        general_exploded: pd.Series = general_tags_series.explode()[
            general_tags_series.explode().notna()
        ].drop_duplicates()  # type: ignore
        tags_with_categories.extend([(tag, "general") for tag in general_exploded])

    if "character" in active_categories:
        character_tags_series = df["tags"].apply(
            lambda x: x.get("character", []) if isinstance(x, dict) else [],
        )
        character_exploded: pd.Series = character_tags_series.explode()[
            character_tags_series.explode().notna()
        ].drop_duplicates()  # type: ignore
        tags_with_categories.extend([(tag, "character") for tag in character_exploded])

    # Create mapping with priority: first occurrence wins
    tag_to_category: dict[str, str] = {}
    for tag, category in tags_with_categories:
        if tag not in tag_to_category:
            tag_to_category[tag] = category

    return tag_to_category


def filter_tags_pandas(
    tag_frequencies: Mapping[str, int],
    base_label_set: set[str],
    tag_categories: dict[str, str],
    selected_categories: Collection[str],
    min_count: int = 1,
    allowed_tags: Collection[str] | None = None,
) -> dict[str, int]:
    """Filter and categorize tags using Pandas DataFrame operations.

    Combines multiple filtering steps in a single vectorized operation instead
    of chaining multiple dictionary comprehensions.

    Args:
        tag_frequencies: Mapping of tags to frequencies (dict or Counter).
        base_label_set: Set of base model tags to exclude.
        tag_categories: Dictionary mapping tag names to category names.
        selected_categories: Categories to include (rating, general, character).
        min_count: Minimum occurrence count for new tags.
        allowed_tags: Optional allow list for new tags.

    Returns:
        Dictionary mapping new tags to their counts.
    """
    # Convert to DataFrame for vectorized operations
    df = pd.DataFrame(
        list(tag_frequencies.items()),
        columns=["tag", "count"],
    )

    # Filter: exclude base tags
    df = df[~df["tag"].isin(base_label_set)]

    # Map categories
    df["category"] = df["tag"].map(tag_categories)

    # Filter: keep selected categories
    df = df[df["category"].isin(selected_categories)]

    # Filter: min count
    df = df[df["count"] >= min_count]

    # Filter: allowed tags if specified
    if allowed_tags is not None:
        df = df[df["tag"].isin(allowed_tags)]

    return dict(zip(df["tag"], df["count"], strict=False))


def create_label_mapping(
    dataset: Dataset,
    categories: Sequence[str] | None = None,
    *,
    min_count: int = 1,
    allowed_tags: Collection[str] | None = None,
) -> tuple[dict[str, int], Counter[str]]:
    """Create label mapping using Pandas for frequency counting.

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

    # Count frequencies with Pandas
    frequencies = count_tag_frequencies(dataset, active_categories)

    # Filter and sort
    filtered_tags = [
        tag
        for tag, count in frequencies.items()
        if count >= min_count and (allowed_tags is None or tag in allowed_tags)
    ]

    sorted_tags = sorted(filtered_tags)
    mapping = {tag: idx for idx, tag in enumerate(sorted_tags)}

    return mapping, frequencies


def categorize_label_list(
    label_list: list[str],
    base_label_indices: dict[str, set[str]],
    dataset_tag_categories: dict[str, str],
) -> dict[str, list[str]]:
    """Categorize labels using Pandas for vectorized operations.

    Splits labels into rating, general, and character categories using
    base model categories first, then dataset categories for new tags.

    Args:
        label_list: Ordered list of all label names.
        base_label_indices: Dict mapping category names to sets of base label names.
            Example: {"rating": {"safe", "explicit"}, "general": {...}, ...}
        dataset_tag_categories: Dict mapping tag names to category names for new tags.

    Returns:
        Dictionary mapping category names to lists of tags.
    """
    # Convert to DataFrame for vectorized operations
    df = pd.DataFrame(
        {"tag": label_list},
    )

    # Check base categories first (priority: rating > character > general)
    def categorize_tag(tag: str) -> str:
        if tag in base_label_indices.get("rating", set()):
            return "rating"
        if tag in base_label_indices.get("character", set()):
            return "character"
        if tag in base_label_indices.get("general", set()):
            return "general"
        # New tags: use dataset category
        return dataset_tag_categories.get(tag, "general")

    df["category"] = df["tag"].apply(categorize_tag)

    # Group tags by category
    result: dict[str, list[str]] = {}
    for category in ["rating", "general", "character"]:
        result[category] = df[df["category"] == category]["tag"].tolist()

    return result


def save_labels_as_csv(
    label_list: list[str],
    tag_categories: dict[str, list[str]],
    output_path: Path,
) -> None:
    """Save labels in WD Tagger v3 compatible CSV format (selected_tags.csv).

    Args:
        label_list: Ordered list of all label names.
        tag_categories: Dictionary mapping category names to lists of tags in that category.
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

    # Write CSV using Pandas
    df = pd.DataFrame(
        {
            "tag_id": list(range(len(label_list))),
            "name": label_list,
            "category": [tag_to_category.get(tag, CATEGORY_MAP["general"]) for tag in label_list],
            "count": [0] * len(label_list),  # Placeholder; not computed from dataset
        },
    )

    df.to_csv(output_path, index=False)


def save_label_mapping_as_json(
    label_mapping: dict[str, int],
    output_path: Path,
) -> None:
    """Save label mapping as JSON for inference compatibility.

    Args:
        label_mapping: Dictionary mapping tag strings to indices.
        output_path: Path to save the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)


def load_label_mapping_from_json(path: Path) -> dict[str, int]:
    """Load label mapping from JSON.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary mapping tag strings to indices.
    """
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def encode_multi_labels(
    tags_dict: Mapping[str, Any],
    label_mapping: dict[str, int],
    categories: Sequence[str],
    *,
    rating_value: Any | None = None,
) -> torch.Tensor:
    """Encode tags as a multi-hot vector respecting category filters.

    Args:
        tags_dict: Dictionary with 'general' and 'character' keys containing tag lists.
        label_mapping: Dictionary mapping tag strings to indices.
        categories: Sequence of categories to include.
        rating_value: Optional rating value to encode.

    Returns:
        Multi-hot encoded tensor of shape (num_labels,).
    """
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

    if "rating" in categories and rating_value is not None:
        normalized = _normalize_rating_value(rating_value)
        if normalized is not None:
            add_tag(normalized)

    return labels


def create_transform_function(
    transform: Callable,
    label_mapping: dict[str, int],
    categories: Sequence[str],
) -> Callable:
    """Create a batched transform function for Dataset.set_transform.

    Transforms are applied on-the-fly without intermediate dataset creation.
    Supports data augmentation that varies per epoch.

    Args:
        transform: Transform function to apply to images (from augmentation.py).
        label_mapping: Dictionary mapping tag strings to indices.
        categories: Sequence of tag categories to encode.

    Returns:
        Transform function compatible with Dataset.set_transform.
    """
    active_categories = _validate_categories(categories)

    def transform_function(examples: dict) -> dict:
        """Apply transforms and encode labels in batch.

        Args:
            examples: Batch of examples from the dataset.

        Returns:
            Transformed examples with pixel_values and labels.
        """
        # Apply image transforms (vectorized if possible)
        images = [transform(img) for img in examples["image"]]
        examples["pixel_values"] = torch.stack(images)

        # Encode multi-labels
        ratings = examples.get("rating", [None] * len(examples["tags"]))
        labels = []

        for i, tags in enumerate(examples["tags"]):
            rating_value = ratings[i] if isinstance(ratings, list) else ratings
            encoded = encode_multi_labels(
                tags,
                label_mapping,
                active_categories,
                rating_value=rating_value,
            )
            labels.append(encoded)

        examples["labels"] = torch.stack(labels)

        # Remove unnecessary fields to save memory
        for key in ["image", "md5", "source", "score", "rating", "tags"]:
            examples.pop(key, None)

        return examples

    return transform_function


def get_dataset_statistics(
    dataset: Dataset,
) -> dict[str, Any]:
    """Compute statistics about the dataset using Pandas vectorized operations.

    Uses zero-copy conversion to Pandas from Arrow.
    All operations use vectorized Pandas methods without Python-level iteration.

    Args:
        dataset: Dataset to analyze.

    Returns:
        Dictionary containing dataset statistics.
    """
    # Convert to Pandas DataFrame (zero-copy from Arrow)
    df: pd.DataFrame = dataset.remove_columns("image").to_pandas()  # type: ignore

    stats: dict[str, Any] = {
        "num_examples": len(df),
        "tag_counts": {"general": 0, "character": 0, "rating": 0},
        "rating_distribution": {},
    }

    # Count general tags (vectorized)
    general_tags_series: pd.Series = df["tags"].apply(  # type: ignore
        lambda x: len(x.get("general", [])) if isinstance(x, dict) else 0,
    )
    stats["tag_counts"]["general"] = int(general_tags_series.sum())

    # Count character tags (vectorized)
    character_tags_series: pd.Series = df["tags"].apply(  # type: ignore
        lambda x: len(x.get("character", [])) if isinstance(x, dict) else 0,
    )
    stats["tag_counts"]["character"] = int(character_tags_series.sum())

    # Count rating tags and distribution (vectorized)
    rating_tags_series: pd.Series = df["rating"].apply(_normalize_rating_value)  # type: ignore
    rating_counts: pd.Series = rating_tags_series.value_counts()  # type: ignore
    stats["tag_counts"]["rating"] = int(rating_counts.sum())
    stats["rating_distribution"] = rating_counts.to_dict()

    return stats
