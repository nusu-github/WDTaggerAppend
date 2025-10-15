"""Compatibility re-exports for augmentation utilities.

This module now forwards to :mod:`wd_tagger_append.custom_processing` so that image
processing logic lives in a single place. The public surface remains unchanged for
callers inside the project.
"""

from wd_tagger_append.custom_processing import (
    AugmentationConfig,
    EnsureRGB,
    PadToSquare,
    RandomCutoutTransform,
    RandomHorizontalFlipTransform,
    RandomRotationTransform,
    RandomSquareCropTransform,
    ResizeWithInterpolationTransform,
    ToBGRTensor,
    build_eval_transform,
    build_train_transform,
)

__all__ = [
    "AugmentationConfig",
    "EnsureRGB",
    "PadToSquare",
    "RandomCutoutTransform",
    "RandomHorizontalFlipTransform",
    "RandomRotationTransform",
    "RandomSquareCropTransform",
    "ResizeWithInterpolationTransform",
    "ToBGRTensor",
    "build_eval_transform",
    "build_train_transform",
]
