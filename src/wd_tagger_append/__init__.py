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
from .constants import RATING_LABELS, LabelCategory
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
from .lora_config import (
    LoRATargetPattern,
    generate_lora_target_modules,
    get_lora_target_pattern,
    get_recommended_block_range,
)
from .model_export import configure_model_for_remote, copy_custom_processor_code
from .prediction import TagPredictionResult

__all__ = [
    "MODEL_REPO_MAP",
    "RATING_LABELS",
    "AugmentationConfig",
    "DatasetSource",
    "EnsureRGB",
    "LabelCategory",
    "LoRATargetPattern",
    "ModelLabels",
    "ModelName",
    "PadToSquare",
    "RandomSquareCropTransform",
    "ResizeWithInterpolationTransform",
    "TagPredictionResult",
    "ToBGRTensor",
    "build_eval_transform",
    "build_train_transform",
    "configure_model_for_remote",
    "copy_custom_processor_code",
    "detect_dataset_source",
    "generate_lora_target_modules",
    "get_lora_target_pattern",
    "get_model_repo_id",
    "get_recommended_block_range",
    "load_dataset_from_hub",
    "load_dataset_with_origin",
    "load_labels_from_hub",
]
