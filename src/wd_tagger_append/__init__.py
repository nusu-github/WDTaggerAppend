"""WD Tagger Append - Tools for model training and inference."""

from .augmentations import (
    AugmentationConfig,
    EnsureRGB,
    PadToSquare,
    RandomSquareCropTransform,
    ResizeWithInterpolationTransform,
    ToBGRTensor,
    build_eval_transform,
    build_train_transform,
)
from .dataset_utils import (
    DatasetSource,
    detect_dataset_source,
    load_dataset_from_hub,
    load_dataset_with_origin,
)
from .labels import (
    MODEL_REPO_MAP,
    ModelLabels,
    ModelName,
    get_model_repo_id,
    load_labels_from_hub,
)
from .model_export import configure_model_for_remote, copy_custom_processor_code

__all__ = [
    "MODEL_REPO_MAP",
    "AugmentationConfig",
    "DatasetSource",
    "EnsureRGB",
    "ModelLabels",
    "ModelName",
    "PadToSquare",
    "RandomSquareCropTransform",
    "ResizeWithInterpolationTransform",
    "ToBGRTensor",
    "build_eval_transform",
    "build_train_transform",
    "configure_model_for_remote",
    "copy_custom_processor_code",
    "detect_dataset_source",
    "get_model_repo_id",
    "load_dataset_from_hub",
    "load_dataset_with_origin",
    "load_labels_from_hub",
]
