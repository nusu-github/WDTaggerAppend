"""Shared utilities for loading WD Tagger models and label metadata."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from transformers import BitsAndBytesConfig

MODEL_REPO_MAP = {
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "vit-large": "SmilingWolf/wd-vit-large-tagger-v3",
    "eva02-large": "SmilingWolf/wd-eva02-large-tagger-v3",
}

CLASSIFIER_SKIP_MODULES: tuple[str, ...] = (
    "head",
    "head.fc",
    "timm_model.head",
    "timm_model.head.fc",
)


@dataclass
class LabelData:
    """Container for label metadata loaded from selected_tags.csv."""

    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def _load_labels_from_csv(csv_path: Path) -> LabelData:
    """Load label metadata from a selected_tags.csv file."""
    df: pd.DataFrame = pd.read_csv(csv_path)
    return LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )


def _resolve_base_model_identifier(model: str, repo_override: str | None) -> str:
    """Resolve the base model repo or path from inputs."""
    return repo_override or MODEL_REPO_MAP.get(model, model)


def _is_local_path(source: str) -> bool:
    """Return True if the given source refers to an existing local path."""
    try:
        return Path(source).expanduser().exists()
    except OSError:
        return False


def _create_quantization_config() -> BitsAndBytesConfig:
    """Construct a BitsAndBytes quantization configuration for 8-bit loading."""
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=list(
            CLASSIFIER_SKIP_MODULES,
        ),
    )


def _parse_precision(precision: str | None) -> torch.dtype | None:
    """Parse precision flag into torch dtype."""
    bf16_supported = torch.cuda.is_bf16_supported()
    if precision is None or precision.lower() == "fp32":
        return None
    normalized = precision.lower()
    if normalized == "bf16":
        if not bf16_supported:
            msg = "bfloat16 is not supported on this device."
            raise ValueError(msg)
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    msg = "Precision must be one of: fp32, bf16, fp16."
    raise ValueError(msg)


def _load_csv_from_source(
    source: str | None,
    adapter: str | None,
    adapter_revision: str | None,
    adapter_token: str | None,
) -> Path | None:
    """Locate a selected_tags.csv file from CLI options or adapter repo."""
    if source is not None:
        path = Path(source).expanduser()
        if not path.exists():
            msg = f"Label file not found: {source}"
            raise FileNotFoundError(msg)
        return path

    if adapter is None:
        return None

    if _is_local_path(adapter):
        candidate = Path(adapter) / "selected_tags.csv"
        return candidate if candidate.exists() else None
    try:
        downloaded = hf_hub_download(
            repo_id=adapter,
            filename="selected_tags.csv",
            revision=adapter_revision,
            token=adapter_token,
        )
        return Path(downloaded)
    except HfHubHTTPError:
        return None


def _load_label_data(
    base_repo: str,
    revision: str | None,
    token: str | None,
    labels_path: Path | None,
    adapter: str | None,
    adapter_revision: str | None,
    adapter_token: str | None,
    fallback_repo: str | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> LabelData:
    """Load label metadata, falling back to the base model's selected_tags.csv."""
    if _is_local_path(base_repo):
        local_csv = Path(base_repo).expanduser() / "selected_tags.csv"
        if local_csv.exists():
            base_labels = _load_labels_from_csv(local_csv)
        elif fallback_repo is not None and not _is_local_path(fallback_repo):
            if warning_callback is not None:
                warning_callback(
                    "selected_tags.csv not found locally; "
                    f"downloading labels from '{fallback_repo}'.",
                )
            base_labels = load_labels_hf(
                repo_id=fallback_repo,
                revision=revision,
                token=token,
            )
        else:
            if warning_callback is not None:
                warning_callback(
                    "Warning: selected_tags.csv not found in local base model directory; "
                    "category data for new labels will default to 'general'.",
                )
            base_labels = LabelData(names=[], rating=[], general=[], character=[])
    else:
        base_labels = load_labels_hf(repo_id=base_repo, revision=revision, token=token)

    csv_path = _load_csv_from_source(
        str(labels_path) if labels_path is not None else None,
        adapter=adapter,
        adapter_revision=adapter_revision,
        adapter_token=adapter_token,
    )
    if csv_path is None:
        return base_labels

    if warning_callback is not None:
        warning_callback(f"Loading labels from {csv_path}...")
    return _load_labels_from_csv(csv_path)


def load_labels_hf(
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> LabelData:
    """Download selected_tags.csv from the Hugging Face Hub."""
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
            token=token,
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as exc:
        msg = f"selected_tags.csv failed to download from {repo_id}"
        raise FileNotFoundError(msg) from exc

    return _load_labels_from_csv(csv_path)


def get_tags(
    probs: torch.Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
) -> tuple[str, str, dict[str, float], dict[str, float], dict[str, float]]:
    """Convert probabilities into caption/tag outputs."""
    prob_values = probs.detach().cpu().numpy()

    rating_indices = np.asarray(labels.rating, dtype=np.int64)
    general_indices = np.asarray(labels.general, dtype=np.int64)
    character_indices = np.asarray(labels.character, dtype=np.int64)

    rating_scores = prob_values[rating_indices]
    rating_sorted_order = np.argsort(rating_scores)[::-1] if rating_indices.size else rating_indices
    rating_ordered_indices = (
        rating_indices[rating_sorted_order] if rating_indices.size else rating_indices
    )
    rating_labels: dict[str, float] = {
        labels.names[int(index)]: float(prob_values[int(index)]) for index in rating_ordered_indices
    }

    general_scores = prob_values[general_indices]
    general_mask = general_scores > gen_threshold
    general_filtered_indices = general_indices[general_mask]
    general_filtered_scores = general_scores[general_mask]
    general_sorted_order = np.argsort(general_filtered_scores)[::-1]
    general_ordered_indices = general_filtered_indices[general_sorted_order]
    gen_labels = {
        labels.names[int(index)]: float(prob_values[int(index)])
        for index in general_ordered_indices
    }

    character_scores = prob_values[character_indices]
    character_mask = character_scores > char_threshold
    character_filtered_indices = character_indices[character_mask]
    character_filtered_scores = character_scores[character_mask]
    character_sorted_order = np.argsort(character_filtered_scores)[::-1]
    character_ordered_indices = character_filtered_indices[character_sorted_order]
    char_labels = {
        labels.names[int(index)]: float(prob_values[int(index)])
        for index in character_ordered_indices
    }

    combined_names = [labels.names[int(index)] for index in general_ordered_indices]
    combined_names.extend(labels.names[int(index)] for index in character_ordered_indices)

    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", r"\(").replace(")", r"\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


__all__ = [
    "MODEL_REPO_MAP",
    "LabelData",
    "_create_quantization_config",
    "_is_local_path",
    "_load_label_data",
    "_parse_precision",
    "_resolve_base_model_identifier",
    "get_tags",
    "load_labels_hf",
]
