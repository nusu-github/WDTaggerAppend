from __future__ import annotations

import torch

from wd_tagger_append.train import (
    TransformPlanner,
    _build_dataset_index_mapping,
    _map_precomputed_labels,
)


class _DummyImageProcessor:
    def __call__(self, *, images, return_tensors, do_train_augmentations):
        tensor = images if isinstance(images, torch.Tensor) else torch.as_tensor(images)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return {"pixel_values": tensor.unsqueeze(0)}


def test_build_dataset_index_mapping_aligns_with_final_label_ids() -> None:
    dataset_mapping = {"cat": 0, "dog": 1, "fox": 2}
    label2id = {"cat": 0, "dog": 1, "wolf": 2}

    mapping = _build_dataset_index_mapping(dataset_mapping, label2id)
    assert mapping == [0, 1, -1]


def test_map_precomputed_labels_maps_known_indices() -> None:
    row = [1, 0, 1, 0]
    mapping = [2, -1, 0, 3]
    result = _map_precomputed_labels(row, mapping, num_labels=5)

    assert result.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_transform_planner_uses_precomputed_label_matrix() -> None:
    processor = _DummyImageProcessor()
    label2id = {"base": 0, "cat": 1, "dog": 2}
    categories = ("rating", "general")
    dataset_index_mapping = [-1, 1, 2]

    planner = TransformPlanner(
        image_processor=processor,  # type: ignore[arg-type]
        label2id=label2id,
        categories=categories,
        dataset_index_mapping=dataset_index_mapping,
        precomputed_column="label_matrix",
    )

    transform = planner.train_transform()
    examples = {
        "image": [torch.zeros(3)],
        "label_matrix": [[0, 1, 0]],
        "tags": [{"general": ["cat"], "character": []}],
        "rating": ["general"],
    }

    output = transform(examples)
    assert "pixel_values" in output
    assert "labels" in output

    labels = output["labels"]
    assert labels.shape == (1, 3)
    assert labels[0, 1].item() == 1.0
    assert labels[0, 2].item() == 0.0
