"""Multilabel stratification helpers shared across CLI workflows."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)

from datasets import Dataset, DatasetDict

from .dataset_utils import (
    LabelMapping,
    LabelMappingBuilder,
    MetadataStream,
    TagCategorySet,
    encode_multi_labels,
)

DEFAULT_STRATIFICATION_SEED = 42


@dataclass(frozen=True)
class LabelMatrixResult:
    """Hold label matrix data alongside metadata required for reuse."""

    matrix: np.ndarray
    mapping: LabelMapping
    categories: TagCategorySet


def build_label_matrix(
    dataset: Dataset,
    categories: TagCategorySet | None = None,
) -> LabelMatrixResult:
    """Create a dense multi-hot label matrix for iterative stratification."""
    effective_categories = categories or TagCategorySet(("general", "character", "rating"))
    mapping = LabelMappingBuilder(effective_categories, min_count=1).build(dataset)

    num_examples = len(dataset)
    num_labels = len(mapping)
    matrix = np.zeros((num_examples, num_labels), dtype=np.int8)

    for row_index, example in enumerate(MetadataStream(dataset)):
        tags = example.get("tags") or {}
        rating_value = example.get("rating")
        encoded = encode_multi_labels(
            tags,
            mapping.label_to_id,
            effective_categories.as_tuple(),
            rating_value=rating_value,
        )
        matrix[row_index] = encoded.to(dtype=torch.int8).numpy()

    return LabelMatrixResult(matrix=matrix, mapping=mapping, categories=effective_categories)


def create_shuffle_splitter(
    test_size: float,
    seed: int = DEFAULT_STRATIFICATION_SEED,
    n_splits: int = 1,
) -> MultilabelStratifiedShuffleSplit:
    """Factory for multilabel-aware train/test shuffle splitters."""
    return MultilabelStratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=seed,
    )


def create_kfold_splitter(
    n_splits: int,
    seed: int = DEFAULT_STRATIFICATION_SEED,
    shuffle: bool = True,
) -> MultilabelStratifiedKFold:
    """Factory for multilabel-aware K-fold splitters."""
    return MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=seed,
    )


def generate_shuffle_split_indices(
    label_matrix: np.ndarray,
    splitter: MultilabelStratifiedShuffleSplit,
) -> tuple[np.ndarray, np.ndarray]:
    """Create train/validation indices using a shuffle splitter."""
    if splitter.n_splits != 1:
        msg = "Shuffle splitter for train/validation must generate exactly one split"
        raise ValueError(msg)

    placeholder = np.zeros((label_matrix.shape[0], 1), dtype=np.int8)
    split_iter = splitter.split(placeholder, label_matrix)
    train_indices, validation_indices = next(split_iter)
    return train_indices, validation_indices


def dataset_from_indices(
    dataset: Dataset,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> DatasetDict:
    """Materialize a DatasetDict from train/validation index lists."""
    return DatasetDict(
        {
            "train": dataset.select(train_indices),
            "validation": dataset.select(validation_indices),
        },
    )


def build_split_metadata(
    splitter_name: str,
    seed: int,
    params: Mapping[str, Any],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    matrix_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Serialize split configuration and indices into JSON-friendly data."""
    metadata: dict[str, Any] = {
        "version": 1,
        "splitter": {
            "name": splitter_name,
            "seed": seed,
            "params": dict(params),
        },
        "splits": {
            "train": [int(index) for index in train_indices],
            "validation": [int(index) for index in validation_indices],
        },
    }
    if matrix_shape is not None:
        metadata["label_matrix"] = {
            "rows": int(matrix_shape[0]),
            "columns": int(matrix_shape[1]),
        }
    return metadata


__all__ = [
    "DEFAULT_STRATIFICATION_SEED",
    "LabelMatrixResult",
    "build_label_matrix",
    "build_split_metadata",
    "create_kfold_splitter",
    "create_shuffle_splitter",
    "dataset_from_indices",
    "generate_shuffle_split_indices",
]
