"""Utilities for preparing models and processors for Hub export."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from transformers import PreTrainedModel

CUSTOM_PROCESSOR_FILENAME: Final[str] = "processing_wd_tagger.py"
_AUTO_MAP_ENTRY: Final[dict[str, str]] = {
    "AutoImageProcessor": "processing_wd_tagger.WDTaggerImageProcessor",
}


def configure_model_for_remote(model: PreTrainedModel) -> None:
    """Configure a model to use the custom image processor when uploaded to the Hub."""
    model.config.problem_type = "multi_label_classification"
    model.config.auto_map = _AUTO_MAP_ENTRY.copy()


def copy_custom_processor_code(destination: Path) -> Path:
    """Copy the local custom processor implementation next to the saved model files."""
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "custom_processing.py"
    if not source.exists():
        msg = f"Custom processor source not found: {source}"
        raise FileNotFoundError(msg)

    target = destination / CUSTOM_PROCESSOR_FILENAME
    shutil.copy(source, target)
    return target
