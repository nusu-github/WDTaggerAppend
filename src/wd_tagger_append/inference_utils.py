"""Inference helpers with focused responsibilities."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

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


@dataclass(frozen=True)
class LabelCatalog:
    """Immutable collection of label names and category indices."""

    names: tuple[str, ...]
    rating_indices: tuple[int, ...]
    general_indices: tuple[int, ...]
    character_indices: tuple[int, ...]

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame) -> Self:
        names = tuple(frame["name"].tolist())
        rating_indices = tuple(frame.index[frame["category"] == 9].tolist())
        general_indices = tuple(frame.index[frame["category"] == 0].tolist())
        character_indices = tuple(frame.index[frame["category"] == 4].tolist())
        return cls(names, rating_indices, general_indices, character_indices)

    @classmethod
    def empty(cls) -> Self:
        return cls((), (), (), ())

    def label_for(self, index: int) -> str:
        return self.names[index]


@dataclass(frozen=True)
class LabelData:
    """Compatibility representation used by existing CLI flows."""

    names: list[str]
    rating: list[int]
    general: list[int]
    character: list[int]

    @classmethod
    def from_catalog(cls, catalog: LabelCatalog) -> Self:
        return cls(
            names=list(catalog.names),
            rating=list(catalog.rating_indices),
            general=list(catalog.general_indices),
            character=list(catalog.character_indices),
        )

    def to_catalog(self) -> LabelCatalog:
        return LabelCatalog(
            tuple(self.names),
            tuple(self.rating),
            tuple(self.general),
            tuple(self.character),
        )


@dataclass(frozen=True)
class ModelIdentifier:
    """Resolve model identifiers while honoring user overrides."""

    value: str
    override: str | None = None

    def resolve(self) -> str:
        if self.override is not None:
            return self.override
        return MODEL_REPO_MAP.get(self.value, self.value)


class LocalPathInspector:
    """Encapsulate local path checks for clarity and testing."""

    @staticmethod
    def exists(source: str) -> bool:
        try:
            return Path(source).expanduser().exists()
        except OSError:
            return False


class QuantizationConfigFactory:
    """Build quantization configs without leaking classifier internals."""

    @staticmethod
    def for_classifier() -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=list(CLASSIFIER_SKIP_MODULES),
        )


class PrecisionParser:
    """Parse precision flags while validating device capabilities."""

    @staticmethod
    def parse(value: str | None) -> torch.dtype | None:
        if value is None or value.lower() == "fp32":
            return None

        normalized = value.lower()
        if normalized == "bf16":
            if not torch.cuda.is_bf16_supported():
                msg = "bfloat16 is not supported on this device."
                raise ValueError(msg)
            return torch.bfloat16
        if normalized == "fp16":
            return torch.float16
        msg = "Precision must be one of: fp32, bf16, fp16."
        raise ValueError(msg)


class LabelCatalogReader:
    """Read persisted label catalogs from CSV files."""

    @staticmethod
    def read_csv(path: Path) -> LabelCatalog:
        frame = pd.read_csv(path)
        return LabelCatalog.from_dataframe(frame)


class LabelHubFetcher:
    """Download label artifacts from the Hugging Face Hub."""

    def __init__(self, downloader: Callable[..., str] = hf_hub_download) -> None:
        self._downloader = downloader

    def download_csv(
        self,
        repo_id: str,
        revision: str | None = None,
        token: str | None = None,
    ) -> Path:
        try:
            csv_path = self._downloader(
                repo_id=repo_id,
                filename="selected_tags.csv",
                revision=revision,
                token=token,
            )
        except HfHubHTTPError as exc:
            msg = f"selected_tags.csv failed to download from {repo_id}"
            raise FileNotFoundError(msg) from exc
        return Path(csv_path).resolve()

    def fetch_catalog(
        self,
        repo_id: str,
        revision: str | None = None,
        token: str | None = None,
    ) -> LabelCatalog:
        csv_path = self.download_csv(repo_id, revision=revision, token=token)
        return LabelCatalogReader.read_csv(csv_path)


class LabelArtifactLocator:
    """Locate adapter-specific label artifacts."""

    def __init__(self, hub_fetcher: LabelHubFetcher) -> None:
        self._hub_fetcher = hub_fetcher

    def locate(
        self,
        explicit_path: Path | None,
        adapter: str | None,
        adapter_revision: str | None,
        adapter_token: str | None,
    ) -> Path | None:
        if explicit_path is not None:
            path = explicit_path.expanduser()
            if not path.exists():
                msg = f"Label file not found: {path}"
                raise FileNotFoundError(msg)
            return path

        if adapter is None:
            return None

        if LocalPathInspector.exists(adapter):
            candidate = Path(adapter).expanduser() / "selected_tags.csv"
            return candidate if candidate.exists() else None

        try:
            return self._hub_fetcher.download_csv(
                adapter,
                revision=adapter_revision,
                token=adapter_token,
            )
        except FileNotFoundError:
            return None


class LabelCatalogLoader:
    """Load label catalogs while honoring adapter overrides and fallbacks."""

    def __init__(
        self,
        hub_fetcher: LabelHubFetcher | None = None,
        warning_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._hub_fetcher = hub_fetcher or LabelHubFetcher()
        self._warning_callback = warning_callback

    def load(
        self,
        base_identifier: str,
        base_revision: str | None,
        base_token: str | None,
        labels_path: Path | None,
        adapter: str | None,
        adapter_revision: str | None,
        adapter_token: str | None,
        fallback_repo: str | None = None,
        forget_base_labels: bool = False,
    ) -> LabelCatalog:
        locator = LabelArtifactLocator(self._hub_fetcher)
        csv_path = locator.locate(
            explicit_path=labels_path,
            adapter=adapter,
            adapter_revision=adapter_revision,
            adapter_token=adapter_token,
        )

        if csv_path is None:
            if forget_base_labels:
                msg = (
                    "No label file found. Provide --labels-path or ensure the adapter contains "
                    "selected_tags.csv when --forget-base-labels is enabled."
                )
                raise FileNotFoundError(msg)

            return self._load_base_catalog(
                base_identifier,
                revision=base_revision,
                token=base_token,
                fallback_repo=fallback_repo,
            )

        if self._warning_callback is not None:
            self._warning_callback(f"Loading labels from {csv_path}...")

        return LabelCatalogReader.read_csv(csv_path)

    def _load_base_catalog(
        self,
        base_identifier: str,
        revision: str | None,
        token: str | None,
        fallback_repo: str | None,
    ) -> LabelCatalog:
        if not LocalPathInspector.exists(base_identifier):
            return self._hub_fetcher.fetch_catalog(base_identifier, revision=revision, token=token)
        local_csv = Path(base_identifier).expanduser() / "selected_tags.csv"
        if local_csv.exists():
            return LabelCatalogReader.read_csv(local_csv)

        if fallback_repo is not None and not LocalPathInspector.exists(fallback_repo):
            if self._warning_callback is not None:
                self._warning_callback(
                    "selected_tags.csv not found locally; "
                    f"downloading labels from '{fallback_repo}'.",
                )
            return self._hub_fetcher.fetch_catalog(
                fallback_repo,
                revision=revision,
                token=token,
            )

        if self._warning_callback is not None:
            self._warning_callback(
                "selected_tags.csv not found in local base model directory; "
                "category data for new labels will default to 'general'.",
            )
        return LabelCatalog.empty()


class ProbabilityFormatter:
    """Transform probability tensors into human-readable tag outputs."""

    def __init__(self, catalog: LabelCatalog) -> None:
        self._catalog = catalog

    def format(
        self,
        probabilities: torch.Tensor,
        general_threshold: float,
        character_threshold: float,
    ) -> tuple[str, str, dict[str, float], dict[str, float], dict[str, float]]:
        values = probabilities.detach().cpu().numpy()

        rating_labels = self._collect_scores(values, self._catalog.rating_indices)
        general_labels = self._collect_scores(
            values,
            self._catalog.general_indices,
            threshold=general_threshold,
        )
        character_labels = self._collect_scores(
            values,
            self._catalog.character_indices,
            threshold=character_threshold,
        )

        caption_parts = list(general_labels)
        caption_parts.extend(character_labels)
        caption = ", ".join(caption_parts)
        tag_string = caption.replace("_", " ").replace("(", r"\(").replace(")", r"\)")

        rating_scores = {name: rating_labels[name] for name in rating_labels}
        char_scores = {name: character_labels[name] for name in character_labels}
        general_scores = {name: general_labels[name] for name in general_labels}

        return caption, tag_string, rating_scores, char_scores, general_scores

    def _collect_scores(
        self,
        values: np.ndarray,
        indices: tuple[int, ...],
        threshold: float | None = None,
    ) -> dict[str, float]:
        if not indices:
            return {}

        index_array = np.asarray(indices, dtype=np.int64)
        scores = values[index_array]

        if threshold is not None:
            mask = scores > threshold
            index_array = index_array[mask]
            scores = scores[mask]

        if not len(index_array):
            return {}

        order = np.argsort(scores)[::-1]
        ordered_indices = index_array[order]

        result: dict[str, float] = {}
        for idx in ordered_indices:
            name = self._catalog.label_for(int(idx))
            result[name] = float(values[int(idx)])
        return result


def load_labels_hf(
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> LabelData:
    """Load label catalogs directly from the Hugging Face Hub."""
    fetcher = LabelHubFetcher()
    catalog = fetcher.fetch_catalog(repo_id, revision=revision, token=token)
    return LabelData.from_catalog(catalog)


def _create_quantization_config() -> BitsAndBytesConfig:
    """Compatibility wrapper returning an 8-bit configuration."""
    return QuantizationConfigFactory.for_classifier()


def _parse_precision(value: str | None) -> torch.dtype | None:
    """Compatibility wrapper delegating to PrecisionParser."""
    return PrecisionParser.parse(value)


def _resolve_base_model_identifier(model: str, repo_override: str | None) -> str:
    """Compatibility wrapper for resolving model identifiers."""
    return ModelIdentifier(model, repo_override).resolve()


def _is_local_path(source: str) -> bool:
    """Compatibility wrapper checking for local resources."""
    return LocalPathInspector.exists(source)


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
    """Compatibility wrapper around LabelCatalogLoader."""
    loader = LabelCatalogLoader(warning_callback=warning_callback)
    catalog = loader.load(
        base_repo,
        base_revision=revision,
        base_token=token,
        labels_path=labels_path,
        adapter=adapter,
        adapter_revision=adapter_revision,
        adapter_token=adapter_token,
        fallback_repo=fallback_repo,
    )
    return LabelData.from_catalog(catalog)


def get_tags(
    probs: torch.Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
) -> tuple[str, str, dict[str, float], dict[str, float], dict[str, float]]:
    """Compatibility wrapper building on ProbabilityFormatter."""
    formatter = ProbabilityFormatter(labels.to_catalog())
    return formatter.format(
        probs,
        general_threshold=gen_threshold,
        character_threshold=char_threshold,
    )


__all__ = [
    "CLASSIFIER_SKIP_MODULES",
    "MODEL_REPO_MAP",
    "LabelArtifactLocator",
    "LabelCatalog",
    "LabelCatalogLoader",
    "LabelCatalogReader",
    "LabelData",
    "LabelHubFetcher",
    "LocalPathInspector",
    "ModelIdentifier",
    "PrecisionParser",
    "ProbabilityFormatter",
    "QuantizationConfigFactory",
    "_create_quantization_config",
    "_is_local_path",
    "_load_label_data",
    "_parse_precision",
    "_resolve_base_model_identifier",
    "get_tags",
    "load_labels_hf",
]
