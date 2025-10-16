"""Shared constants and enumerations for WD Tagger Append."""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class LabelCategory(IntEnum):
    """Numeric categories used in tag CSV files."""

    GENERAL = 0
    CHARACTER = 4
    RATING = 9


RATING_LABELS: Final[tuple[str, ...]] = ("general", "sensitive", "questionable", "explicit")

# Filenames that the project reads or writes frequently.
DEFAULT_TAGS_FILENAME: Final[str] = "selected_tags.csv"
CUSTOM_PROCESSOR_FILENAME: Final[str] = "processing_wd_tagger.py"
