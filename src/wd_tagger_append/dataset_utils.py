"""Dataset processing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image,
    IterableDataset,
    IterableDatasetDict,
    Sequence as DatasetSequence,
    Value,
)
from wd_tagger_append.augmentations import AugmentationConfig, build_train_transform
from wd_tagger_append.constants import RATING_LABELS
from wd_tagger_append.custom_processing import resolve_interpolation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from numpy.typing import NDArray
    from transformers import TimmWrapperImageProcessor

    from .labels import ModelLabels


class ImageLabel(TypedDict):
    """Label information for a single image."""

    md5: str
    image: str
    rating: str
    tags_general: list[str]
    tags_character: list[str]


class DatasetTagFields(TypedDict):
    """Type definition for tag fields within a dataset item."""

    tags_general: list[str]
    tags_character: list[str]


class CollatorInputItem(TypedDict):
    """Type definition for items processed by the collate function."""

    pixel_values: torch.Tensor
    labels: NDArray[np.float32]


class DatasetSource(str, Enum):
    """Enumeration of supported dataset origins."""

    LOCAL = "local"
    HUB = "hub"


BASE_DATASET_FEATURES = Features(
    {
        "md5": Value("string"),
        "image": Image(),
        "rating": ClassLabel(names=list(RATING_LABELS)),
        "tags_general": DatasetSequence(Value("string")),
        "tags_character": DatasetSequence(Value("string")),
    },
)


@dataclass(slots=True, frozen=True)
class ProcessorConfig:
    """Extracted image processor configuration used for dataset transforms."""

    size: tuple[int, int]
    interpolation: InterpolationMode
    mean: torch.Tensor
    std: torch.Tensor


def _extract_processor_config(processor: TimmWrapperImageProcessor) -> ProcessorConfig:
    """Extract relevant configuration from an image processor."""
    processor_height, processor_width = get_image_processor_size(processor)

    interpolation_attr = getattr(processor, "_eval_interpolation", None)
    if isinstance(interpolation_attr, InterpolationMode):
        interpolation = interpolation_attr
    else:
        data_config = getattr(processor, "data_config", {})
        interpolation = resolve_interpolation(str(data_config.get("interpolation", "bicubic")))

    mean_tensor = getattr(processor, "_bgr_mean", None)
    std_tensor = getattr(processor, "_bgr_std", None)
    if isinstance(mean_tensor, torch.Tensor) and isinstance(std_tensor, torch.Tensor):
        bgr_mean = mean_tensor.clone().detach()
        bgr_std = std_tensor.clone().detach()
    else:
        data_config = getattr(processor, "data_config", {})
        mean_values = data_config.get("mean", (0.5, 0.5, 0.5))
        std_values = data_config.get("std", (0.5, 0.5, 0.5))
        mean_t = torch.tensor(mean_values, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(std_values, dtype=torch.float32).view(3, 1, 1)
        bgr_mean = mean_t.flip(0)
        bgr_std = std_t.flip(0)

    return ProcessorConfig(
        size=(processor_height, processor_width),
        interpolation=interpolation,
        mean=bgr_mean,
        std=bgr_std,
    )


def collect_image_paths(folder_path: str) -> list[str]:
    """Recursively collect image paths from a folder.

    Args:
        folder_path: Path to the folder to search

    Returns:
        List of image file paths
    """
    image_extensions = PILImage.registered_extensions().keys()
    folder = Path(folder_path)
    return [
        str(p.resolve())
        for p in folder.rglob("*")
        if p.suffix.lower() in image_extensions and p.is_file()
    ]


def parse_image_labels(image_paths: Sequence[str]) -> Iterator[ImageLabel]:
    """Yield label information from JSON files with the same name as image paths.

    Args:
        image_paths: List of image file paths

    Returns:
        Iterator of dictionaries containing label information
    """
    for image_path in image_paths:
        path = Path(image_path)
        label_file = path.with_suffix(path.suffix + ".json")
        if not label_file.is_file():
            continue

        try:
            with open(label_file, encoding="utf-8") as f:
                json_data = json.load(f)

            rating_code = json_data["rating"]
            tags_general = json_data["tags_general"]
            tags_character = json_data["tags_character"]
            md5 = json_data["md5"]

            # Convert rating code to full name
            rating_map = {"g": "general", "s": "sensitive", "q": "questionable", "e": "explicit"}
            rating = rating_map.get(rating_code)
            if rating is None:
                print(f"Unknown rating '{rating_code}' in {image_path}")
                continue

            yield {
                "md5": md5,
                "image": image_path,
                "rating": rating,
                "tags_general": tags_general,
                "tags_character": tags_character,
            }
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            continue


def _normalize_rating_value(rating: int | str) -> str:
    """Return the canonical rating label."""
    if isinstance(rating, str):
        return rating
    try:
        return RATING_LABELS[rating]
    except (IndexError, TypeError) as error:
        msg = f"Unsupported rating value: {rating!r}"
        raise ValueError(msg) from error


def create_label_encoding_function(
    labels: ModelLabels,
) -> Callable[[dict[str, list[Any]]], dict[str, list[Any]]]:
    """Create a batched map function that converts tags into multi-hot label vectors."""
    name_to_index = {name: idx for idx, name in enumerate(labels.names)}
    rating_indices = [int(index) for index in labels.rating_indices]
    if len(rating_indices) < len(RATING_LABELS):
        msg = "Expected four rating indices (general, sensitive, questionable, explicit)"
        raise ValueError(msg)
    rating_map: dict[str, int] = dict(zip(RATING_LABELS, rating_indices, strict=True))
    num_labels = labels.num_labels

    def encode_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        tags_general_list = cast("list[list[str]]", batch["tags_general"])
        tags_character_list = cast("list[list[str]]", batch["tags_character"])
        ratings = batch["rating"]

        encoded_labels: list[list[bool]] = []
        for general_tags, character_tags, rating in zip(
            tags_general_list,
            tags_character_list,
            ratings,
            strict=False,
        ):
            label_vector = [False] * num_labels

            for tag in general_tags:
                label_index = name_to_index[tag]
                label_vector[label_index] = True

            for tag in character_tags:
                label_index = name_to_index[tag]
                label_vector[label_index] = True

            rating_label = _normalize_rating_value(rating)
            label_vector[rating_map[rating_label]] = True
            encoded_labels.append(label_vector)

        return {"labels": encoded_labels}

    return encode_batch


def load_dataset_from_folder(
    folder_path: str,
) -> Dataset:
    """Load images and labels from a folder, removing duplicates to create a Dataset.

    Args:
        folder_path: Path to the dataset folder

    Returns:
        Hugging Face Dataset
    """
    image_paths = collect_image_paths(folder_path)
    seen_md5: set[str] = set()

    def sample_generator() -> Iterator[ImageLabel]:
        for item in parse_image_labels(image_paths):
            md5 = item["md5"]
            if md5 in seen_md5:
                continue
            seen_md5.add(md5)
            yield item

    return cast("Dataset", Dataset.from_generator(sample_generator, features=BASE_DATASET_FEATURES))


def detect_dataset_source(dataset_source: str) -> DatasetSource:
    """Determine whether the provided source points to a local directory or Hub repository."""
    path = Path(dataset_source).expanduser()
    if path.exists() and path.is_dir():
        return DatasetSource.LOCAL
    return DatasetSource.HUB


def load_dataset_with_origin(
    dataset_source: str,
    *,
    token: str | None = None,
    revision: str | None = None,
    source: DatasetSource | None = None,
) -> tuple[DatasetSource, Dataset]:
    """Load a dataset and return both the materialised dataset and its origin."""
    actual_source = source or detect_dataset_source(dataset_source)
    if actual_source is DatasetSource.LOCAL:
        dataset = load_dataset_from_folder(str(Path(dataset_source).expanduser()))
    else:
        dataset = load_dataset_from_hub(
            repo_id=dataset_source,
            token=token,
            revision=revision,
        )
    return actual_source, dataset


def analyze_new_tags(dataset: Dataset, labels: ModelLabels) -> tuple[set[str], set[str]]:
    """Analyze new tags in the dataset.

    Args:
        dataset: Dataset to analyze
        labels: Existing label information

    Returns:
        Tuple of (set of new general tags, set of new character tags)
    """
    new_general_tags = set()
    new_character_tags = set()

    # Ensure the image column is not decoded for faster processing
    dataset = dataset.cast_column("image", Image(decode=False))

    for raw_item in tqdm(dataset, desc="Analyzing tags"):
        item = cast("DatasetTagFields", raw_item)
        general_tags = item["tags_general"]
        for tag in general_tags:
            if tag not in labels.names:
                new_general_tags.add(tag)

        character_tags = item["tags_character"]
        for tag in character_tags:
            if tag not in labels.names:
                new_character_tags.add(tag)

    return new_general_tags, new_character_tags


def get_image_processor_size(
    processor: TimmWrapperImageProcessor,
) -> tuple[int, int]:
    """Extract the target image size from a Hugging Face image processor."""
    # Try data_config first (for TimmWrapperImageProcessor)
    if hasattr(processor, "data_config") and isinstance(processor.data_config, dict):
        input_size = processor.data_config.get("input_size")
        if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
            # Format is [channels, height, width]
            return int(input_size[1]), int(input_size[2])

    size: dict[str, int] | Sequence[int] | int | None = getattr(
        processor,
        "crop_size",
        None,
    ) or getattr(processor, "size", None)

    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
        if len(size) == 1:
            value = int(next(iter(size.values())))
            return value, value
    if isinstance(size, (list, tuple)):
        if len(size) >= 2:
            return int(size[0]), int(size[1])
        if not size:
            msg = "Processor size sequence is empty."
            raise ValueError(msg)
        value = int(size[0])
        return value, value
    if isinstance(size, int):
        return size, size

    msg = "Unable to infer image size from processor configuration."
    raise ValueError(msg)


def create_transform_function(
    labels: ModelLabels,
    processor: TimmWrapperImageProcessor,
    *,
    config: AugmentationConfig,
) -> Callable[[dict[str, list[Any]]], dict[str, list[Any]]]:
    """Create a dataset transformation function.

    Args:
        labels: Label information
        processor: Image processor
        config: Augmentation configuration

    Returns:
        Dataset transformation function
    """
    processor_config = _extract_processor_config(processor)
    if config.size != processor_config.size:
        config = replace(config, size=processor_config.size)

    interpolation_candidates = (
        None if config.random_interpolation else (processor_config.interpolation,)
    )
    transform_pipeline = build_train_transform(
        config,
        default_interpolation=processor_config.interpolation,
        mean=processor_config.mean,
        std=processor_config.std,
        interpolation_candidates=interpolation_candidates,
    )

    expected_num_labels = labels.num_labels

    def transform_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        processed_images = [transform_pipeline(pil_image) for pil_image in batch["image"]]
        batch["pixel_values"] = processed_images

        if "labels" not in batch:
            msg = "Expected 'labels' column to be present before applying set_transform"
            raise KeyError(msg)

        processed_labels: list[NDArray[np.float32]] = []
        for label_vector in cast("list[bool]", batch["labels"]):
            label_array = np.asarray(label_vector, dtype=bool)
            if label_array.shape != (expected_num_labels,):
                msg = (
                    "Label vector has unexpected shape "
                    f"{label_array.shape}; expected ({expected_num_labels!s},)"
                )
                raise ValueError(msg)
            label_array = label_array.astype(np.float32)
            processed_labels.append(label_array)

        batch["labels"] = processed_labels

        batch.pop("md5", None)
        batch.pop("image", None)
        batch.pop("tags_general", None)
        batch.pop("tags_character", None)
        batch.pop("rating", None)

        return batch

    return transform_batch


def create_collate_fn(
    *,
    mixup_alpha: float | None = None,
    mixup_prob: float = 1.0,
) -> Callable[[list[CollatorInputItem]], dict[str, torch.Tensor]]:
    """Create a collate function with optional MixUp augmentation."""

    def collate(examples: list[CollatorInputItem]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor(
            np.stack([example["labels"] for example in examples]),
            dtype=torch.float32,
        )

        if (
            mixup_alpha is not None
            and mixup_alpha > 0
            and len(examples) > 1
            and mixup_prob > 0
            and torch.rand(1).item() <= mixup_prob
        ):
            beta_distribution = torch.distributions.Beta(mixup_alpha, mixup_alpha)
            lambda_values = beta_distribution.sample((pixel_values.size(0),))
            perm = torch.randperm(pixel_values.size(0))

            lambda_images = lambda_values.view(-1, 1, 1, 1)
            lambda_labels = lambda_values.view(-1, 1)

            pixel_values = lambda_images * pixel_values + (1 - lambda_images) * pixel_values[perm]
            labels = lambda_labels * labels + (1 - lambda_labels) * labels[perm]

        return {"pixel_values": pixel_values, "labels": labels}

    return collate


def collate_fn(examples: list[CollatorInputItem]) -> dict[str, torch.Tensor]:
    """Default collate function without MixUp (backwards compatibility)."""
    return create_collate_fn()(examples)


def load_dataset_from_hub(
    repo_id: str,
    token: str | None = None,
    revision: str | None = None,
) -> Dataset:
    """Load a dataset from Hugging Face Hub.

    Args:
        repo_id: Hugging Face dataset repository ID
        token: Authentication token (optional)
        revision: Dataset revision/branch (optional)

    Returns:
        Loaded Dataset
    """
    from datasets import load_dataset

    dataset = load_dataset(
        repo_id,
        split="train",
        token=token,
        revision=revision,
    )
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        dataset = dataset["train"]

    if isinstance(dataset, IterableDataset):
        msg = "Iterable datasets are not supported for training pipelines."
        raise TypeError(msg)

    if not isinstance(dataset, Dataset):
        msg = f"Expected a Dataset object, but got {type(dataset)}."
        raise TypeError(msg)

    # Remove the _base_model column if it exists (metadata column)
    if "_base_model" in dataset.column_names:
        dataset = dataset.remove_columns(["_base_model"])

    return dataset
