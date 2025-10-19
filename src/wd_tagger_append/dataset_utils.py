"""Dataset utility functions for WD Tagger training.

This module provides utilities for label encoding, dataset transformation,
and integration with training pipelines using standard library primitives.
"""

import csv
import json
import numbers
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from datasets import ClassLabel, Dataset, Features

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

    if invalid := [category for category in categories if category not in VALID_TAG_CATEGORIES]:
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


def _iter_metadata(dataset: Dataset | Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Yield metadata dictionaries without materializing large intermediate objects."""
    if isinstance(dataset, Dataset):
        columns_to_drop = [col for col in ("image",) if col in dataset.column_names]
        working_dataset = dataset.remove_columns(columns_to_drop) if columns_to_drop else dataset

        rating_feature = None
        features = getattr(working_dataset, "features", None)
        if isinstance(features, (Features, Mapping)):
            rating_feature = features.get("rating")

        for record in working_dataset:
            example = cast("Mapping[str, Any]", record)
            if isinstance(rating_feature, ClassLabel):
                rating_value = example.get("rating")
                if isinstance(rating_value, numbers.Integral):
                    example = dict(example)
                    example["rating"] = rating_feature.int2str(int(rating_value))
            yield example
        return

    for record in dataset:
        yield cast("Mapping[str, Any]", record)


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
    return RATING_CODE_TO_NAME.get(candidate, candidate) if candidate else None


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
    dataset: Dataset | Iterable[Mapping[str, Any]],
    categories: Sequence[str] | None = None,
) -> Counter[str]:
    """Count tag occurrences using Counter instead of Pandas."""
    active_categories = _validate_categories(categories)
    counts: Counter[str] = Counter()

    for example in _iter_metadata(dataset):
        tags = example.get("tags") or {}

        if "general" in active_categories:
            counts.update(tag for tag in tags.get("general", []) if tag)

        if "character" in active_categories:
            counts.update(tag for tag in tags.get("character", []) if tag)

        if "rating" in active_categories:
            if normalized := _normalize_rating_value(example.get("rating")):
                counts.update((normalized,))

    return counts


def determine_tag_categories(
    dataset: Dataset | Iterable[Mapping[str, Any]],
    categories: Sequence[str] | None = None,
) -> dict[str, str]:
    """Determine which category each tag belongs to without repeated Pandas operations."""
    active_categories = _validate_categories(categories)

    rating_tags: set[str] = set()
    general_tags: set[str] = set()
    character_tags: set[str] = set()

    for example in _iter_metadata(dataset):
        tags = example.get("tags") or {}

        if "rating" in active_categories:
            if normalized := _normalize_rating_value(example.get("rating")):
                rating_tags.add(normalized)

        if "general" in active_categories:
            general_tags.update(tag for tag in tags.get("general", []) if tag)

        if "character" in active_categories:
            character_tags.update(tag for tag in tags.get("character", []) if tag)

    tag_to_category: dict[str, str] = {}
    for tag_set, category in (
        (rating_tags, "rating"),
        (general_tags, "general"),
        (character_tags, "character"),
    ):
        if category not in active_categories:
            continue
        for tag in tag_set:
            tag_to_category.setdefault(tag, category)

    return tag_to_category


def filter_tags_pandas(
    tag_frequencies: Mapping[str, int],
    base_label_set: set[str],
    tag_categories: dict[str, str],
    selected_categories: Collection[str],
    min_count: int = 1,
    allowed_tags: Collection[str] | None = None,
) -> dict[str, int]:
    """Filter tags using plain dictionaries for clarity and safety."""
    if min_count < 1:
        return {}

    allowed_lookup = set(allowed_tags) if allowed_tags is not None else None
    selected_lookup = set(selected_categories)

    result: dict[str, int] = {}
    for tag, count in tag_frequencies.items():
        if tag in base_label_set:
            continue
        category = tag_categories.get(tag)
        if category not in selected_lookup:
            continue
        if count < min_count:
            continue
        if allowed_lookup is not None and tag not in allowed_lookup:
            continue
        result[tag] = count
    return result


def create_label_mapping(
    dataset: Dataset | Iterable[Mapping[str, Any]],
    categories: Sequence[str] | None = None,
    *,
    min_count: int = 1,
    allowed_tags: Collection[str] | None = None,
) -> tuple[dict[str, int], Counter[str]]:
    """Create label mapping using Counter for frequency counting."""
    if min_count < 1:
        msg = "min_count must be >= 1"
        raise ValueError(msg)

    active_categories = _validate_categories(categories)
    frequencies = count_tag_frequencies(dataset, active_categories)

    allowed_lookup = set(allowed_tags) if allowed_tags is not None else None

    filtered_tags = [
        tag
        for tag, count in frequencies.items()
        if count >= min_count and (allowed_lookup is None or tag in allowed_lookup)
    ]

    sorted_tags = sorted(filtered_tags)
    mapping = {tag: idx for idx, tag in enumerate(sorted_tags)}

    return mapping, frequencies


def categorize_label_list(
    label_list: list[str],
    base_label_indices: dict[str, set[str]],
    dataset_tag_categories: dict[str, str],
) -> dict[str, list[str]]:
    """Categorize labels without relying on Pandas."""
    result: dict[str, list[str]] = {"rating": [], "general": [], "character": []}

    rating_base = base_label_indices.get("rating", set())
    character_base = base_label_indices.get("character", set())
    general_base = base_label_indices.get("general", set())

    for tag in label_list:
        if tag in rating_base:
            category = "rating"
        elif tag in character_base:
            category = "character"
        elif tag in general_base:
            category = "general"
        else:
            category = dataset_tag_categories.get(tag, "general")
        result.setdefault(category, []).append(tag)

    return result


def save_labels_as_csv(
    label_list: list[str],
    tag_categories: dict[str, list[str]],
    output_path: Path,
) -> None:
    """Save labels in WD Tagger v3 compatible CSV format (selected_tags.csv)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tag_to_category: dict[str, int] = {}
    for category_name, tags in tag_categories.items():
        category_code = CATEGORY_MAP.get(category_name)
        if category_code is None:
            continue
        for tag in tags:
            tag_to_category[tag] = category_code

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["tag_id", "name", "category", "count"])
        writer.writeheader()
        for idx, tag in enumerate(label_list):
            writer.writerow(
                {
                    "tag_id": idx,
                    "name": tag,
                    "category": tag_to_category.get(tag, CATEGORY_MAP["general"]),
                    "count": 0,
                },
            )


def save_label_mapping_as_json(
    label_mapping: dict[str, int],
    output_path: Path,
) -> None:
    """Save label mapping as JSON for inference compatibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)


def load_label_mapping_from_json(path: Path) -> dict[str, int]:
    """Load label mapping from JSON."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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
    """Create a batched transform function for Dataset.set_transform."""
    active_categories = _validate_categories(categories)

    def transform_function(examples: dict) -> dict:
        images = [transform(img) for img in examples["image"]]
        examples["pixel_values"] = torch.stack(images)

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

        for key in ["image", "md5", "source", "score", "rating", "tags"]:
            examples.pop(key, None)

        return examples

    return transform_function


def get_dataset_statistics(
    dataset: Dataset | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute statistics about the dataset without relying on Pandas."""
    num_examples = 0
    total_general = 0
    total_character = 0
    rating_counts: Counter[str] = Counter()

    for example in _iter_metadata(dataset):
        num_examples += 1
        tags = example.get("tags") or {}
        total_general += len(tags.get("general", []) or [])
        total_character += len(tags.get("character", []) or [])
        if normalized := _normalize_rating_value(example.get("rating")):
            rating_counts.update((normalized,))

    stats: dict[str, Any] = {
        "num_examples": num_examples,
        "tag_counts": {
            "general": total_general,
            "character": total_character,
            "rating": int(sum(rating_counts.values())),
        },
        "rating_distribution": dict(rating_counts),
    }

    return stats
