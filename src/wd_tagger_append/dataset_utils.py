"""Dataset and label utilities expressed as composable services."""

import csv
import json
import numbers
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import torch

from datasets import ClassLabel, Dataset, Features

VALID_TAG_CATEGORIES: tuple[str, ...] = ("rating", "general", "character")

RATING_CODE_TO_NAME = {
    "g": "general",
    "s": "sensitive",
    "q": "questionable",
    "e": "explicit",
}

CATEGORY_CODE_MAP = {
    "rating": 9,
    "general": 0,
    "character": 4,
}


@dataclass(frozen=True)
class TagCategorySet:
    """Validated category selection used across dataset operations."""

    values: tuple[str, ...] = ("general", "character")

    def __post_init__(self) -> None:
        unique: list[str] = []
        for category in self.values:
            if category not in VALID_TAG_CATEGORIES:
                msg = f"Unsupported tag category: {category}"
                raise ValueError(msg)
            if category not in unique:
                unique.append(category)
        object.__setattr__(self, "values", tuple(unique))

    @classmethod
    def from_optional(cls, categories: Sequence[str] | None) -> Self:
        """Create a validated selection from an optional sequence."""
        return cls() if categories is None else cls(tuple(categories))

    def includes(self, category: str) -> bool:
        """Return True when the category is part of the selection."""
        return category in self.values

    def as_tuple(self) -> tuple[str, ...]:
        """Expose the selection as an immutable tuple."""
        return self.values


class MetadataStream:
    """Iterate over dataset metadata while shielding callers from storage details."""

    def __init__(self, dataset: Dataset | Iterable[Mapping[str, Any]]) -> None:
        self._dataset = dataset

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        if isinstance(self._dataset, Dataset):
            yield from self._iterate_dataset(self._dataset)
            return
        for record in self._dataset:
            yield cast("Mapping[str, Any]", record)

    def _iterate_dataset(self, dataset: Dataset) -> Iterator[Mapping[str, Any]]:
        columns_to_drop = [column for column in ("image",) if column in dataset.column_names]
        working_set = dataset.remove_columns(columns_to_drop) if columns_to_drop else dataset

        rating_feature = None
        features = getattr(working_set, "features", None)
        if isinstance(features, (Features, Mapping)):
            rating_feature = features.get("rating")

        for record in working_set:
            example = cast("Mapping[str, Any]", record)
            if isinstance(rating_feature, ClassLabel):
                rating_value = example.get("rating")
                if isinstance(rating_value, numbers.Integral):
                    mutable = dict(example)
                    mutable["rating"] = rating_feature.int2str(int(rating_value))
                    example = mutable
            yield example


class RatingNormalizer:
    """Normalize rating values coming from various dataset representations."""

    @staticmethod
    def normalize(value: Any) -> str | None:
        if value is None:
            return None

        if isinstance(value, numbers.Integral):
            return None

        if isinstance(value, (list, tuple)):
            for item in value:
                normalized = RatingNormalizer.normalize(item)
                if normalized is not None:
                    return normalized
            return None

        if not isinstance(value, str):
            return None

        candidate = value.strip()
        return RATING_CODE_TO_NAME.get(candidate, candidate) if candidate else None


@dataclass(frozen=True)
class AllowedTagSpec:
    """Allowed tag definitions plus optional replacement mapping."""

    values: frozenset[str]
    replacements: dict[str, str]

    def __post_init__(self) -> None:
        missing: set[str] = {tag for tag in self.replacements if tag not in self.values}
        if missing:
            msg = f"Replacement keys must be part of the allowed set: {sorted(missing)}"
            raise ValueError(msg)

    def is_empty(self) -> bool:
        return not self.values

    def as_collection(self) -> Collection[str]:
        return self.values

    def replacement_for(self, tag: str) -> str | None:
        return self.replacements.get(tag)


class AllowedTagLoader:
    """Load, sanitize, and expose allow lists for tag filtering."""

    @staticmethod
    def load(path: Path) -> AllowedTagSpec:
        allowed: list[str] = []
        replacements: dict[str, str] = {}

        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue

                original = row[0].strip()
                if not original or original.startswith("#"):
                    continue

                lower_original = original.lower()
                second = row[1].strip() if len(row) > 1 else ""
                if lower_original in {"original", "source", "tag"} and second.lower() in {
                    "replacement",
                    "target",
                    "alias",
                }:
                    continue

                allowed.append(original)
                replacement = second.strip()
                if replacement:
                    replacements[original] = replacement

        return AllowedTagSpec(frozenset(allowed), replacements)


class TagFrequencyCounter:
    """Count tag occurrences without committing to a specific backend."""

    def __init__(self, categories: TagCategorySet) -> None:
        self._categories = categories

    def collect(self, dataset: Dataset | Iterable[Mapping[str, Any]]) -> Counter[str]:
        counts: Counter[str] = Counter()
        stream = MetadataStream(dataset)

        for example in stream:
            tags = example.get("tags") or {}

            if self._categories.includes("general"):
                counts.update(tag for tag in tags.get("general", []) if tag)

            if self._categories.includes("character"):
                counts.update(tag for tag in tags.get("character", []) if tag)

            if self._categories.includes("rating"):
                rating_tag = RatingNormalizer.normalize(example.get("rating"))
                if rating_tag is not None:
                    counts.update((rating_tag,))

        return counts


class TagCategoryResolver:
    """Resolve category assignments for dataset tags."""

    def __init__(self, categories: TagCategorySet) -> None:
        self._categories = categories

    def resolve(self, dataset: Dataset | Iterable[Mapping[str, Any]]) -> dict[str, str]:
        rating_tags: set[str] = set()
        general_tags: set[str] = set()
        character_tags: set[str] = set()

        for example in MetadataStream(dataset):
            tags = example.get("tags") or {}

            if self._categories.includes("rating"):
                rating_value = RatingNormalizer.normalize(example.get("rating"))
                if rating_value is not None:
                    rating_tags.add(rating_value)

            if self._categories.includes("general"):
                general_tags.update(tag for tag in tags.get("general", []) if tag)

            if self._categories.includes("character"):
                character_tags.update(tag for tag in tags.get("character", []) if tag)

        tag_to_category: dict[str, str] = {}
        for tag_set, category in (
            (rating_tags, "rating"),
            (general_tags, "general"),
            (character_tags, "character"),
        ):
            if not self._categories.includes(category):
                continue
            for tag in tag_set:
                tag_to_category.setdefault(tag, category)

        return tag_to_category


class TagFilter:
    """Filter tag counts into an allow-listed subset."""

    def __init__(
        self,
        categories: TagCategorySet,
        *,
        min_count: int = 1,
        allowed_tags: Collection[str] | None = None,
    ) -> None:
        if min_count < 1:
            msg = "min_count must be >= 1"
            raise ValueError(msg)

        self._categories = categories
        self._min_count = min_count
        self._allowed_lookup = set(allowed_tags) if allowed_tags is not None else None

    def apply(
        self,
        tag_frequencies: Mapping[str, int],
        base_labels: Collection[str],
        dataset_tag_categories: Mapping[str, str],
    ) -> dict[str, int]:
        base_lookup = set(base_labels)
        selected_categories = set(self._categories.as_tuple())

        filtered: dict[str, int] = {}
        for tag, count in tag_frequencies.items():
            if tag in base_lookup:
                continue
            category = dataset_tag_categories.get(tag)
            if category not in selected_categories:
                continue
            if count < self._min_count:
                continue
            if self._allowed_lookup is not None and tag not in self._allowed_lookup:
                continue
            filtered[tag] = count

        return filtered


@dataclass(frozen=True)
class LabelMapping:
    """Immutable mapping from labels to indices with helper utilities."""

    label_to_id: dict[str, int]
    frequencies: Counter[str]

    def __post_init__(self) -> None:
        sorted_items = sorted(self.label_to_id.items(), key=lambda item: item[1])
        for expected_index, (_, index) in enumerate(sorted_items):
            if index != expected_index:
                msg = "label_to_id must contain contiguous indices starting at zero"
                raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.label_to_id)

    def labels_in_order(self) -> list[str]:
        return [label for label, _ in sorted(self.label_to_id.items(), key=lambda item: item[1])]

    def categorize(
        self,
        base_label_indices: Mapping[str, Collection[str]],
        dataset_tag_categories: Mapping[str, str],
    ) -> dict[str, list[str]]:
        partitions: dict[str, list[str]] = {"rating": [], "general": [], "character": []}

        base_lookup = {
            category: set(values)
            for category, values in base_label_indices.items()
            if category in partitions
        }

        for label in self.labels_in_order():
            if label in base_lookup.get("rating", set()):
                partitions["rating"].append(label)
                continue
            if label in base_lookup.get("character", set()):
                partitions["character"].append(label)
                continue
            if label in base_lookup.get("general", set()):
                partitions["general"].append(label)
                continue
            category = dataset_tag_categories.get(label, "general")
            partitions.setdefault(category, []).append(label)

        return partitions

    def save_as_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.label_to_id, handle, indent=2, ensure_ascii=False)

    def save_as_csv(self, label_categories: Mapping[str, Sequence[str]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tag_to_category: dict[str, int] = {}
        for category_name, tags in label_categories.items():
            category_code = CATEGORY_CODE_MAP.get(category_name)
            if category_code is None:
                continue
            for tag in tags:
                tag_to_category[tag] = category_code

        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["tag_id", "name", "category", "count"])
            writer.writeheader()
            for idx, tag in enumerate(self.labels_in_order()):
                writer.writerow(
                    {
                        "tag_id": idx,
                        "name": tag,
                        "category": tag_to_category.get(tag, CATEGORY_CODE_MAP["general"]),
                        "count": 0,
                    },
                )


class LabelMappingBuilder:
    """Assemble label mappings from dataset statistics."""

    def __init__(
        self,
        categories: TagCategorySet,
        *,
        min_count: int = 1,
        allowed_tags: Collection[str] | None = None,
    ) -> None:
        if min_count < 1:
            msg = "min_count must be >= 1"
            raise ValueError(msg)

        self._categories = categories
        self._min_count = min_count
        self._allowed_lookup = set(allowed_tags) if allowed_tags is not None else None

    def build(self, dataset: Dataset | Iterable[Mapping[str, Any]]) -> LabelMapping:
        counter = TagFrequencyCounter(self._categories)
        frequencies = counter.collect(dataset)

        filtered_tags = [
            tag
            for tag, count in frequencies.items()
            if count >= self._min_count
            and (self._allowed_lookup is None or tag in self._allowed_lookup)
        ]

        sorted_tags = sorted(filtered_tags)
        mapping = {tag: idx for idx, tag in enumerate(sorted_tags)}
        return LabelMapping(mapping, frequencies)


class LabelMappingReader:
    """Read label mappings from persisted artifacts."""

    @staticmethod
    def load_json(path: Path) -> dict[str, int]:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            msg = "JSON label mapping must be an object"
            raise ValueError(msg)
        return cast("dict[str, int]", data)


class MultiHotEncoder:
    """Encode tag dictionaries into multi-hot tensors."""

    def __init__(self, mapping: LabelMapping, categories: TagCategorySet) -> None:
        self._mapping = mapping
        self._categories = categories

    def __call__(
        self,
        tags: Mapping[str, Any],
        *,
        rating_value: Any | None = None,
    ) -> torch.Tensor:
        num_classes = len(self._mapping)
        encoded = torch.zeros(num_classes, dtype=torch.float32)

        def add_tag(tag: str | None) -> None:
            if tag is None:
                return
            index = self._mapping.label_to_id.get(tag)
            if index is not None:
                encoded[index] = 1.0

        if self._categories.includes("general"):
            for tag in tags.get("general", []):
                add_tag(tag)

        if self._categories.includes("character"):
            for tag in tags.get("character", []):
                add_tag(tag)

        if self._categories.includes("rating") and rating_value is not None:
            normalized = RatingNormalizer.normalize(rating_value)
            if normalized is not None:
                add_tag(normalized)

        return encoded


class TransformBuilder:
    """Create dataset transform functions coupling image transforms and encoders."""

    def __init__(
        self,
        image_transform: Callable[[Any], torch.Tensor],
        encoder: MultiHotEncoder,
    ) -> None:
        self._image_transform = image_transform
        self._encoder = encoder

    def build(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def _transform(examples: dict[str, Any]) -> dict[str, Any]:
            images = [self._image_transform(image) for image in examples["image"]]
            examples["pixel_values"] = torch.stack(images)

            ratings = examples.get("rating")
            batch_size = len(images)

            if isinstance(ratings, list):
                rating_iterable: Iterable[Any | None] = ratings
            else:
                rating_iterable = [ratings] * batch_size

            encoded_labels = [
                self._encoder(tags, rating_value=rating)
                for tags, rating in zip(examples["tags"], rating_iterable, strict=False)
            ]

            examples["labels"] = torch.stack(encoded_labels)

            for key in ("image", "md5", "source", "score", "rating", "tags"):
                examples.pop(key, None)

            return examples

        return _transform


class DatasetStatisticsCollector:
    """Collect simple aggregate statistics for reporting."""

    def __init__(self, dataset: Dataset | Iterable[Mapping[str, Any]]) -> None:
        self._dataset = dataset

    def collect(self) -> dict[str, Any]:
        num_examples = 0
        total_general = 0
        total_character = 0
        rating_counts: Counter[str] = Counter()

        for example in MetadataStream(self._dataset):
            num_examples += 1
            tags = example.get("tags") or {}
            total_general += len(tags.get("general", []) or [])
            total_character += len(tags.get("character", []) or [])
            rating_tag = RatingNormalizer.normalize(example.get("rating"))
            if rating_tag is not None:
                rating_counts.update((rating_tag,))

        return {
            "num_examples": num_examples,
            "tag_counts": {
                "general": total_general,
                "character": total_character,
                "rating": int(sum(rating_counts.values())),
            },
            "rating_distribution": dict(rating_counts),
        }


def load_allowed_tags(path: Path) -> AllowedTagSpec:
    """Compatibility wrapper around AllowedTagLoader."""
    return AllowedTagLoader.load(path)


def apply_replacements_to_counts(
    tag_counts: Mapping[str, int],
    replacements: Mapping[str, str],
) -> dict[str, int]:
    """Map tag counts through replacement definitions."""
    if not replacements:
        return dict(tag_counts)

    remapped: dict[str, int] = {}
    for tag, count in tag_counts.items():
        target = replacements.get(tag, tag)
        remapped[target] = remapped.get(target, 0) + count
    return remapped


def add_replacement_categories(
    categories: Mapping[str, str],
    replacements: Mapping[str, str],
) -> dict[str, str]:
    """Return category assignments augmented with replacement entries."""
    if not replacements:
        return dict(categories)

    updated = dict(categories)
    for original, replacement in replacements.items():
        category = categories.get(original)
        if category is None:
            continue
        updated.setdefault(replacement, category)
    return updated


def count_tag_frequencies(
    dataset: Dataset | Iterable[Mapping[str, Any]],
    categories: Sequence[str] | None = None,
) -> Counter[str]:
    """Count tag frequencies using the TagFrequencyCounter service."""
    selection = TagCategorySet.from_optional(categories)
    counter = TagFrequencyCounter(selection)
    return counter.collect(dataset)


def determine_tag_categories(
    dataset: Dataset | Iterable[Mapping[str, Any]],
    categories: Sequence[str] | None = None,
) -> dict[str, str]:
    """Determine tag categories via TagCategoryResolver."""
    selection = TagCategorySet.from_optional(categories)
    resolver = TagCategoryResolver(selection)
    return resolver.resolve(dataset)


def filter_tags_pandas(
    tag_frequencies: Mapping[str, int],
    base_label_set: set[str],
    tag_categories: Mapping[str, str],
    selected_categories: Collection[str],
    min_count: int = 1,
    allowed_tags: Collection[str] | None = None,
) -> dict[str, int]:
    """Filter tags by delegating to TagFilter."""
    selection = TagCategorySet(tuple(selected_categories))
    flt = TagFilter(selection, min_count=min_count, allowed_tags=allowed_tags)
    return flt.apply(tag_frequencies, base_label_set, tag_categories)


def create_label_mapping(
    dataset: Dataset | Iterable[Mapping[str, Any]],
    categories: Sequence[str] | None = None,
    *,
    min_count: int = 1,
    allowed_tags: Collection[str] | None = None,
) -> tuple[dict[str, int], Counter[str]]:
    """Construct a label mapping using LabelMappingBuilder."""
    selection = TagCategorySet.from_optional(categories)
    builder = LabelMappingBuilder(selection, min_count=min_count, allowed_tags=allowed_tags)
    mapping = builder.build(dataset)
    return mapping.label_to_id, mapping.frequencies


def categorize_label_list(
    label_list: Sequence[str],
    base_label_indices: Mapping[str, Collection[str]],
    dataset_tag_categories: Mapping[str, str],
) -> dict[str, list[str]]:
    """Categorize labels using LabelMapping helper methods."""
    mapping = {label: idx for idx, label in enumerate(label_list)}
    label_mapping = LabelMapping(mapping, Counter())
    return label_mapping.categorize(base_label_indices, dataset_tag_categories)


def save_labels_as_csv(
    label_list: Sequence[str],
    tag_categories: Mapping[str, Sequence[str]],
    output_path: Path,
) -> None:
    """Persist labels to CSV via LabelMapping."""
    mapping = LabelMapping({label: idx for idx, label in enumerate(label_list)}, Counter())
    mapping.save_as_csv(tag_categories, output_path)


def save_label_mapping_as_json(label_mapping: Mapping[str, int], output_path: Path) -> None:
    """Persist label mapping dictionary to JSON."""
    LabelMapping(dict(label_mapping), Counter()).save_as_json(output_path)


def load_label_mapping_from_json(path: Path) -> dict[str, int]:
    """Load label mapping JSON via LabelMappingReader."""
    return LabelMappingReader.load_json(path)


def encode_multi_labels(
    tags_dict: Mapping[str, Any],
    label_mapping: Mapping[str, int],
    categories: Sequence[str],
    *,
    rating_value: Any | None = None,
) -> torch.Tensor:
    """Encode a tag dictionary by composing MultiHotEncoder."""
    mapping = LabelMapping(dict(label_mapping), Counter())
    encoder = MultiHotEncoder(mapping, TagCategorySet(tuple(categories)))
    return encoder(tags_dict, rating_value=rating_value)


def create_transform_function(
    transform: Callable[[Any], torch.Tensor],
    label_mapping: Mapping[str, int],
    categories: Sequence[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a dataset transform function through TransformBuilder."""
    mapping = LabelMapping(dict(label_mapping), Counter())
    encoder = MultiHotEncoder(mapping, TagCategorySet(tuple(categories)))
    builder = TransformBuilder(transform, encoder)
    return builder.build()


def get_dataset_statistics(
    dataset: Dataset | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collect dataset statistics using DatasetStatisticsCollector."""
    collector = DatasetStatisticsCollector(dataset)
    return collector.collect()


__all__ = [
    "AllowedTagLoader",
    "AllowedTagSpec",
    "DatasetStatisticsCollector",
    "LabelMapping",
    "LabelMappingBuilder",
    "LabelMappingReader",
    "MetadataStream",
    "MultiHotEncoder",
    "RatingNormalizer",
    "TagCategoryResolver",
    "TagCategorySet",
    "TagFilter",
    "TagFrequencyCounter",
    "TransformBuilder",
    "add_replacement_categories",
    "apply_replacements_to_counts",
    "categorize_label_list",
    "count_tag_frequencies",
    "create_label_mapping",
    "create_transform_function",
    "determine_tag_categories",
    "encode_multi_labels",
    "filter_tags_pandas",
    "get_dataset_statistics",
    "load_allowed_tags",
    "load_label_mapping_from_json",
    "save_label_mapping_as_json",
    "save_labels_as_csv",
]
