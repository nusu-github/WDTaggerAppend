from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Self, cast

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as F
from transformers.image_processing_base import BatchFeature
from transformers.image_transforms import to_pil_image
from transformers.image_utils import ImageInput, make_flat_list_of_images
from transformers.models.timm_wrapper import TimmWrapperImageProcessor

if TYPE_CHECKING:
    from transformers.utils.generic import TensorType


class EnsureRGB(v2.Transform):
    """Convert images to three-channel RGB tensors with alpha compositing."""

    def transform(self, inpt: Any, params: dict[str, Any]) -> torch.Tensor:
        if isinstance(inpt, torch.Tensor):
            tensor_image = F.to_image(inpt)
        else:
            image = cast("Image.Image", inpt)
            if image.mode not in ["RGB", "RGBA"]:
                has_transparency = "transparency" in image.info
                image = image.convert("RGBA") if has_transparency else image.convert("RGB")
            tensor_image = F.to_image(image)

        if tensor_image.shape[0] == 4:
            alpha = F.to_dtype(tensor_image[3:4], torch.float32, True)
            rgb = F.to_dtype(tensor_image[:3], torch.float32, True)
            alpha = alpha.expand_as(rgb)
            composite = rgb * alpha + (1.0 - alpha)
            tensor_image = composite
        elif tensor_image.shape[0] == 1:
            tensor_image = tensor_image.expand(3, -1, -1)
        elif tensor_image.shape[0] != 3:
            msg = f"Unsupported channel count: {tensor_image.shape[0]}"
            raise ValueError(msg)
        tensor_image = tensor_image[:3]
        return F.to_dtype(
            tensor_image,
            torch.float32,
            not tensor_image.dtype.is_floating_point,
        )


class PadToSquare(v2.Transform):
    """Pad an image tensor to a square canvas using a white background."""

    def __init__(self, fill: int = 255) -> None:
        super().__init__()
        self.fill = int(fill)

    def transform(self, inpt: Any, params: dict[str, Any]) -> torch.Tensor:
        tensor = inpt if isinstance(inpt, torch.Tensor) else F.to_image(inpt)
        tensor = F.to_dtype(tensor, torch.float32, not tensor.dtype.is_floating_point)
        _, height, width = tensor.shape
        max_dim = int(max(height, width))
        pad_left = (max_dim - width) // 2
        pad_right = max_dim - width - pad_left
        pad_top = (max_dim - height) // 2
        pad_bottom = max_dim - height - pad_top
        if pad_left or pad_right or pad_top or pad_bottom:
            fill_value = float(self.fill)
            if fill_value > 1.0:
                fill_value /= 255.0
            tensor = F.pad(tensor, [pad_left, pad_top, pad_right, pad_bottom], fill=fill_value)
        return tensor


class RandomSquareCropTransform(v2.Transform):
    """Crop a random square area from the image."""

    def __init__(self, min_scale: float, max_scale: float) -> None:
        super().__init__()
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        if not flat_inputs:
            return {"crop": False}

        tensor = F.to_image(flat_inputs[0])
        _, height, width = tensor.shape
        base_size = min(width, height)
        if base_size <= 1:
            return {"crop": False}

        min_scale = max(self.min_scale, 0.0)
        max_scale = min(self.max_scale, 1.0)
        if max_scale <= 0 or max_scale <= min_scale:
            return {"crop": False}

        area_scale = random.uniform(min_scale, max_scale)
        if math.isclose(area_scale, 1.0, rel_tol=1e-3):
            return {"crop": False}

        crop_size = max(1, round(base_size * math.sqrt(area_scale)))
        if crop_size >= base_size:
            return {"crop": False}

        max_offset = base_size - crop_size
        left = random.randint(0, max_offset)
        top = random.randint(0, max_offset)
        return {"crop": True, "top": top, "left": left, "size": crop_size}

    def transform(self, inpt: Any, params: dict[str, Any]) -> torch.Tensor:
        tensor = inpt if isinstance(inpt, torch.Tensor) else F.to_image(inpt)
        tensor = F.to_dtype(tensor, torch.float32, not tensor.dtype.is_floating_point)
        if not params.get("crop", False):
            return tensor
        return F.crop(tensor, params["top"], params["left"], params["size"], params["size"])


class ResizeWithInterpolationTransform(v2.Transform):
    """Resize images to a target size with optional interpolation randomisation."""

    def __init__(
        self,
        size: tuple[int, int],
        *,
        randomize: bool,
        default_interpolation: InterpolationMode,
        candidates: tuple[InterpolationMode, ...] | None = None,
    ) -> None:
        super().__init__()
        self.size = (int(size[0]), int(size[1]))
        self.randomize = randomize
        self.default_interpolation = default_interpolation
        self.candidates = candidates if candidates is not None else _INTERPOLATION_METHODS

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        if self.randomize and self.candidates:
            interpolation = random.choice(self.candidates)
        else:
            interpolation = self.default_interpolation
        return {"interpolation": interpolation}

    def transform(self, inpt: Any, params: dict[str, Any]) -> torch.Tensor:
        tensor = inpt if isinstance(inpt, torch.Tensor) else F.to_image(inpt)
        tensor = F.to_dtype(tensor, torch.float32, not tensor.dtype.is_floating_point)
        interpolation = params.get("interpolation", self.default_interpolation)
        target_height, target_width = self.size

        if interpolation == InterpolationMode.LANCZOS:
            # Lanczos resizing does not support pytorch tensors directly
            pil_img = F.to_pil_image(
                F.to_dtype(tensor, torch.uint8, True).contiguous(),
            )
            pil_img = pil_img.resize(
                (target_width, target_height),
                resample=Image.Resampling.LANCZOS,
            )
            resized = F.to_image(pil_img)
            return F.to_dtype(resized, torch.float32, True)

        resized_tensor = F.resize(
            tensor,
            size=[target_height, target_width],
            interpolation=interpolation,
            antialias=True,
        )
        return F.to_dtype(resized_tensor, torch.float32, False)


class ToBGRTensor(v2.Transform):
    """Convert tensors to float, swap channels to BGR, and normalise."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def transform(self, inpt: Any, params: dict[str, Any]) -> torch.Tensor:
        tensor = inpt if isinstance(inpt, torch.Tensor) else F.to_image(inpt)
        tensor = F.to_dtype(
            tensor,
            torch.float32,
            not tensor.dtype.is_floating_point,
        ).contiguous()
        if tensor.size(0) == 3:
            tensor = tensor[[2, 1, 0], :, :]
            mean = cast("torch.Tensor", self.mean)
            std = cast("torch.Tensor", self.std)
            tensor = (tensor - mean) / std
        return tensor


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for per-sample augmentations."""

    size: tuple[int, int]
    apply_flip: bool = True
    flip_prob: float = 0.5
    apply_random_crop: bool = True
    random_crop_min_scale: float = 0.87
    random_crop_max_scale: float = 0.998
    apply_rotation: bool = True
    max_rotation_degrees: float = 45.0
    apply_cutout: bool = True
    cutout_prob: float = 0.5
    cutout_min_ratio: float = 0.05
    cutout_max_ratio: float = 0.35
    random_interpolation: bool = True

    def evaluation_variant(self) -> Self:
        """Return a copy with stochastic augmentations disabled."""
        return type(self)(
            size=self.size,
            apply_flip=False,
            flip_prob=self.flip_prob,
            apply_random_crop=False,
            random_crop_min_scale=self.random_crop_min_scale,
            random_crop_max_scale=self.random_crop_max_scale,
            apply_rotation=False,
            max_rotation_degrees=self.max_rotation_degrees,
            apply_cutout=False,
            cutout_prob=0.0,
            cutout_min_ratio=self.cutout_min_ratio,
            cutout_max_ratio=self.cutout_max_ratio,
            random_interpolation=False,
        )


_INTERPOLATION_METHODS: tuple[InterpolationMode, ...] = (
    InterpolationMode.BILINEAR,
    InterpolationMode.BICUBIC,
    InterpolationMode.LANCZOS,
)

_INTERPOLATION_LOOKUP: dict[str, InterpolationMode] = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def resolve_interpolation(value: str | InterpolationMode) -> InterpolationMode:
    """Resolve a string or interpolation mode into a torchvision interpolation enum."""
    if isinstance(value, InterpolationMode):
        return value
    return _INTERPOLATION_LOOKUP.get(value.lower(), InterpolationMode.BICUBIC)


def build_eval_transform(
    config: AugmentationConfig,
    *,
    interpolation: InterpolationMode,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> v2.Transform:
    """Create an evaluation transform pipeline."""
    transforms: list[v2.Transform] = [
        EnsureRGB(),
        PadToSquare(),
        ResizeWithInterpolationTransform(
            config.size,
            randomize=False,
            default_interpolation=interpolation,
            candidates=(interpolation,),
        ),
        ToBGRTensor(mean, std),
    ]
    return v2.Compose(transforms)


def build_train_transform(
    config: AugmentationConfig,
    *,
    default_interpolation: InterpolationMode,
    mean: torch.Tensor,
    std: torch.Tensor,
    interpolation_candidates: tuple[InterpolationMode, ...] | None = None,
) -> v2.Transform:
    """Create a training transform pipeline based on the augmentation config."""
    candidates = (
        interpolation_candidates if interpolation_candidates is not None else _INTERPOLATION_METHODS
    )

    transforms: list[v2.Transform] = [EnsureRGB(), PadToSquare()]
    if config.apply_flip:
        transforms.append(v2.RandomHorizontalFlip(p=config.flip_prob))
    if config.apply_random_crop:
        transforms.append(
            RandomSquareCropTransform(
                config.random_crop_min_scale,
                config.random_crop_max_scale,
            ),
        )
    if config.apply_rotation:
        transforms.append(
            v2.RandomRotation(
                degrees=(-config.max_rotation_degrees, config.max_rotation_degrees),
                interpolation=InterpolationMode.BILINEAR,
                expand=True,
                fill=1.0,
            ),
        )
    transforms.append(
        ResizeWithInterpolationTransform(
            config.size,
            randomize=config.random_interpolation,
            default_interpolation=default_interpolation,
            candidates=candidates,
        ),
    )
    if config.apply_cutout:
        # Convert cutout_min_ratio and cutout_max_ratio to scale parameter for RandomErasing
        # RandomErasing uses area proportion, so we need to square the ratio
        min_scale = config.cutout_min_ratio**2
        max_scale = config.cutout_max_ratio**2
        transforms.append(
            v2.RandomErasing(
                p=config.cutout_prob,
                scale=(min_scale, max_scale),
                ratio=(1.0, 1.0),  # Square region
                value=127 / 255.0,  # Normalized gray value
            ),
        )
    transforms.append(ToBGRTensor(mean, std))
    return v2.Compose(transforms)


class WDTaggerImageProcessor(TimmWrapperImageProcessor):
    """Custom image processor for WD Tagger models.

    This processor applies padding to make the image square and converts it to BGR format,
    which is expected by the WaifuDiffusion Tagger models.
    """

    def __init__(
        self,
        pretrained_cfg: dict[str, Any],
        architecture: str | None = None,
        *,
        train_augmentation_config: dict[str, Any] | None = None,
        evaluation_augmentation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise processor with custom train/validation transforms."""
        super().__init__(pretrained_cfg=pretrained_cfg, architecture=architecture, **kwargs)

        self.image_height, self.image_width = self._resolve_target_size()
        self._mean = torch.tensor(
            self.data_config.get("mean", (0.5, 0.5, 0.5)),
            dtype=torch.float32,
        ).view(3, 1, 1)
        self._std = torch.tensor(
            self.data_config.get("std", (0.5, 0.5, 0.5)),
            dtype=torch.float32,
        ).view(3, 1, 1)
        self._eval_interpolation = self._resolve_interpolation(
            self.data_config.get("interpolation", "bicubic"),
        )
        self._bgr_mean = self._mean.flip(0)
        self._bgr_std = self._std.flip(0)

        default_train_config = AugmentationConfig(size=(self.image_height, self.image_width))
        if train_augmentation_config is not None:
            coerced = self._coerce_config_dict(train_augmentation_config)
            default_train_config = AugmentationConfig(**coerced)

        default_eval_config = (
            AugmentationConfig(**self._coerce_config_dict(evaluation_augmentation_config))
            if evaluation_augmentation_config is not None
            else default_train_config.evaluation_variant()
        )

        self.train_augmentation_config = default_train_config
        self.eval_augmentation_config = default_eval_config

        self.train_transforms = self._build_train_transform(self.train_augmentation_config)
        self.val_transforms = self._build_eval_transform(self.eval_augmentation_config)

        self._not_supports_tensor_input = True

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: str | TensorType | None = "pt",
    ) -> BatchFeature:
        """Preprocess an image or batch of images.

        Args:
            images: Image to preprocess. Expects a single or batch of images
            return_tensors: The type of tensors to return. Must be 'pt'.
        """
        if return_tensors != "pt":
            msg = (
                f"return_tensors for WDTaggerImageProcessor must be 'pt', but got {return_tensors}"
            )
            raise ValueError(msg)

        image_list = cast("list[Any]", make_flat_list_of_images(images))
        processed_tensors = [self.val_transforms(self._ensure_pil_image(img)) for img in image_list]

        batch = torch.stack(processed_tensors)
        return BatchFeature({"pixel_values": batch}, tensor_type=return_tensors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_pil_image(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            with Image.open(image) as img:
                return img.convert("RGBA") if "transparency" in img.info else img.convert("RGB")
        return to_pil_image(image)

    def _resolve_target_size(self) -> tuple[int, int]:
        input_size = self.data_config.get("input_size")
        if isinstance(input_size, (list, tuple)) and len(input_size) >= 3:
            return int(input_size[-2]), int(input_size[-1])

        crop_size = self.data_config.get("crop_size")
        if isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
            return int(crop_size[0]), int(crop_size[1])
        if isinstance(crop_size, dict) and {"height", "width"} <= crop_size.keys():
            return int(crop_size["height"]), int(crop_size["width"])

        size = self.data_config.get("input_size", (3, 448, 448))
        if isinstance(size, int):
            return int(size), int(size)
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return int(size[0]), int(size[1])

        msg = "Unable to resolve target size from data configuration."
        raise ValueError(msg)

    def _resolve_interpolation(self, value: str) -> InterpolationMode:
        return resolve_interpolation(value)

    def _coerce_config_dict(self, config: dict[str, Any]) -> dict[str, Any]:
        coerced = dict(config)
        if "size" in coerced:
            size_value = coerced["size"]
            if isinstance(size_value, (list, tuple)):
                if len(size_value) != 2:
                    msg = "Augmentation size must contain exactly two values."
                    raise ValueError(msg)
                coerced["size"] = (int(size_value[0]), int(size_value[1]))
        return coerced

    def _build_eval_transform(
        self,
        config: AugmentationConfig,
    ) -> v2.Transform:
        return build_eval_transform(
            config,
            interpolation=self._eval_interpolation,
            mean=self._bgr_mean,
            std=self._bgr_std,
        )

    def _build_train_transform(
        self,
        config: AugmentationConfig,
    ) -> v2.Transform:
        return build_train_transform(
            config,
            default_interpolation=self._eval_interpolation,
            mean=self._bgr_mean,
            std=self._bgr_std,
        )

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()

        # Add serializable augmentation configs
        data["train_augmentation_config"] = asdict(self.train_augmentation_config)
        data["evaluation_augmentation_config"] = asdict(self.eval_augmentation_config)

        # Remove non-serializable attributes
        non_serializable_keys = [
            "_mean",
            "_std",
            "_bgr_mean",
            "_bgr_std",
            "_eval_interpolation",
            "train_transforms",
            "val_transforms",
            "train_augmentation_config",  # Remove the object version
            "eval_augmentation_config",  # Remove the object version
        ]
        for key in non_serializable_keys:
            data.pop(key, None)

        return data
