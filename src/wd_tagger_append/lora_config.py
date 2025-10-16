"""Model-specific LoRA target module configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

    from wd_tagger_append.labels import ModelName


@dataclass(frozen=True)
class LoRATargetPattern:
    """Pattern description for generating LoRA target module names."""

    attention_patterns: tuple[str, ...]
    mlp_patterns: tuple[str, ...]
    block_format: str
    container_name: str | None = None
    container_block_counts: tuple[int, ...] | None = None


_LORA_TARGET_CONFIGS: Final[dict[ModelName, LoRATargetPattern]] = {
    "eva02-large": LoRATargetPattern(
        attention_patterns=("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj"),
        mlp_patterns=("mlp.fc1_g", "mlp.fc1_x", "mlp.fc2"),
        block_format="blocks.{block_idx}",
    ),
    "convnext": LoRATargetPattern(
        attention_patterns=(),
        mlp_patterns=("mlp.fc1", "mlp.fc2"),
        block_format="stages.{stage_idx}.blocks.{block_idx}",
        container_name="stage_idx",
        container_block_counts=(3, 3, 27, 3),
    ),
    "swinv2": LoRATargetPattern(
        attention_patterns=("attn.qkv", "attn.proj"),
        mlp_patterns=("mlp.fc1", "mlp.fc2"),
        block_format="layers.{layer_idx}.blocks.{block_idx}",
        container_name="layer_idx",
        container_block_counts=(2, 2, 18, 2),
    ),
    "vit": LoRATargetPattern(
        attention_patterns=("attn.qkv", "attn.proj"),
        mlp_patterns=("mlp.fc1", "mlp.fc2"),
        block_format="blocks.{block_idx}",
    ),
    "vit-large": LoRATargetPattern(
        attention_patterns=("attn.qkv", "attn.proj"),
        mlp_patterns=("mlp.fc1", "mlp.fc2"),
        block_format="blocks.{block_idx}",
    ),
}


_TOTAL_BLOCKS: Final[dict[ModelName, int]] = {
    "eva02-large": 24,
    "convnext": 36,  # 3 + 3 + 27 + 3 stage blocks
    "swinv2": 24,  # 2 + 2 + 18 + 2 layer blocks
    "vit": 12,
    "vit-large": 24,
}


_RECOMMENDED_BLOCK_RANGES: Final[dict[ModelName, tuple[int, int]]] = {
    "eva02-large": (8, 24),
    "convnext": (6, 36),  # Target stages 2 and 3 (flattened indices)
    "swinv2": (4, 24),  # Target layers 2 and 3 (flattened indices)
    "vit": (6, 12),
    "vit-large": (12, 24),
}


def get_lora_target_pattern(model_name: ModelName) -> LoRATargetPattern:
    """Retrieve the LoRA target pattern for the requested model."""
    try:
        return _LORA_TARGET_CONFIGS[model_name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        valid_models = ", ".join(sorted(_LORA_TARGET_CONFIGS))
        msg = f"No LoRA configuration for model: {model_name}. Valid models: {valid_models}"
        raise ValueError(msg) from exc


def _iter_block_prefixes(
    pattern: LoRATargetPattern,
    start_block: int,
    end_block: int,
) -> Iterable[str]:
    """Yield fully formatted block prefixes based on the pattern description."""
    if pattern.container_block_counts is None:
        for block_idx in range(start_block, end_block):
            yield pattern.block_format.format(block_idx=block_idx)
        return

    if pattern.container_name is None:  # pragma: no cover - construction bug guard
        msg = "container_block_counts provided without container_name."
        raise ValueError(msg)

    counts = pattern.container_block_counts
    assert counts is not None  # Narrow type for the type checker

    container_idx = 0
    remaining = start_block
    while container_idx < len(counts) and remaining >= counts[container_idx]:
        remaining -= counts[container_idx]
        container_idx += 1

    if container_idx == len(counts):
        msg = f"start_block {start_block} exceeds available blocks."
        raise ValueError(msg)

    local_block_idx = remaining
    produced = start_block

    while produced < end_block:
        if container_idx >= len(counts):
            msg = f"end_block {end_block} exceeds available blocks."
            raise ValueError(msg)

        block_count = counts[container_idx]
        while local_block_idx < block_count and produced < end_block:
            yield pattern.block_format.format(
                **{pattern.container_name: container_idx},
                block_idx=local_block_idx,
            )
            local_block_idx += 1
            produced += 1

        container_idx += 1
        local_block_idx = 0


def generate_lora_target_modules(
    model_name: ModelName,
    start_block: int,
    end_block: int,
) -> list[str]:
    """Create a list of LoRA module names for the requested block span."""
    if start_block < 0:
        msg = "start_block must be non-negative."
        raise ValueError(msg)
    if end_block <= start_block:
        msg = "end_block must be greater than start_block."
        raise ValueError(msg)

    total_blocks = _TOTAL_BLOCKS.get(model_name)
    if total_blocks is not None and end_block > total_blocks:
        msg = f"end_block {end_block} exceeds total blocks {total_blocks} for {model_name}."
        raise ValueError(msg)

    pattern = get_lora_target_pattern(model_name)
    targets: list[str] = []

    for block_prefix in _iter_block_prefixes(pattern, start_block, end_block):
        for attn_pattern in pattern.attention_patterns:
            targets.append(f"{block_prefix}.{attn_pattern}")
        for mlp_pattern in pattern.mlp_patterns:
            targets.append(f"{block_prefix}.{mlp_pattern}")

    return targets


def get_recommended_block_range(model_name: ModelName) -> tuple[int, int]:
    """Return the recommended LoRA block range for the specified model."""
    try:
        return _RECOMMENDED_BLOCK_RANGES[model_name]
    except KeyError as exc:
        valid_models = ", ".join(sorted(_RECOMMENDED_BLOCK_RANGES))
        msg = f"No recommended block range for model: {model_name}. Valid models: {valid_models}"
        raise ValueError(msg) from exc
