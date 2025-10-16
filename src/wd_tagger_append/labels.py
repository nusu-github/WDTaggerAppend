"""Label management utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Self, TypeAlias, cast

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from wd_tagger_append.constants import DEFAULT_TAGS_FILENAME, LabelCategory

if TYPE_CHECKING:
    from pandas._typing import UsecolsArgType

ModelName: TypeAlias = Literal["convnext", "eva02-large", "swinv2", "vit-large", "vit"]

# Model repository mapping
MODEL_REPO_MAP: Final[dict[ModelName, str]] = {
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "eva02-large": "SmilingWolf/wd-eva02-large-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "vit-large": "SmilingWolf/wd-vit-large-tagger-v3",
    "vit": "SmilingWolf/wd-vit-tagger-v3",
}


@dataclass
class ModelLabels:
    """Data class for managing model label information."""

    names: list[str]
    rating_indices: list[int]
    general_indices: list[int]
    character_indices: list[int]

    @property
    def num_labels(self) -> int:
        """Return the total number of labels."""
        return len(self.names)

    def to_label_mappings(self) -> tuple[dict[str, int], dict[int, str]]:
        """Generate label ID mappings.

        Returns:
            Tuple of (label2id, id2label)
        """
        label2id = {name: i for i, name in enumerate(self.names)}
        id2label = {i: name for name, i in label2id.items()}
        return label2id, id2label

    def extend_with_new_tags(
        self,
        new_general_tags: set[str],
        new_character_tags: set[str],
    ) -> Self:
        """Create a new ModelLabels with new tags added.

        Args:
            new_general_tags: General tags to add
            new_character_tags: Character tags to add

        Returns:
            Extended ModelLabels instance
        """
        # Create an instance of the same runtime class as `self` for proper `Self` typing.
        clazz: type[Self] = type(self)
        extended = clazz(
            names=self.names.copy(),
            rating_indices=self.rating_indices.copy(),
            general_indices=self.general_indices.copy(),
            character_indices=self.character_indices.copy(),
        )

        # Add new general tags
        for tag in sorted(new_general_tags):
            index = len(extended.names)
            extended.names.append(tag)
            extended.general_indices.append(index)

        # Add new character tags
        for tag in sorted(new_character_tags):
            index = len(extended.names)
            extended.names.append(tag)
            extended.character_indices.append(index)

        return extended


def labels_to_dataframe(labels: ModelLabels) -> pd.DataFrame:
    """Convert a ModelLabels object to a pandas DataFrame."""
    category_map: dict[int, int] = {}
    for index in labels.rating_indices:
        category_map[int(index)] = LabelCategory.RATING.value
    for index in labels.general_indices:
        category_map[int(index)] = LabelCategory.GENERAL.value
    for index in labels.character_indices:
        category_map[int(index)] = LabelCategory.CHARACTER.value

    rows: list[dict[str, object]] = []
    for index, name in enumerate(labels.names):
        category = category_map.get(index)
        if category is None:
            msg = f"Missing category mapping for label index {index}: {name}"
            raise ValueError(msg)
        rows.append({"name": name, "category": category})

    return pd.DataFrame(rows)


def load_labels_from_hub(
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> ModelLabels:
    """Load label information for pretrained models from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID or local path
        revision: Revision (optional)
        token: Authentication token (optional)

    Returns:
        ModelLabels instance

    Raises:
        FileNotFoundError: If selected_tags.csv is not found
    """
    tags_filename = DEFAULT_TAGS_FILENAME
    local_path = Path(repo_id)
    if local_path.is_dir() and (local_path / tags_filename).exists():
        csv_path = (local_path / tags_filename).resolve()
    else:
        try:
            csv_path = hf_hub_download(
                repo_id=repo_id,
                filename=tags_filename,
                revision=revision,
                token=token,
            )
            csv_path = Path(csv_path).resolve()
        except HfHubHTTPError as e:
            msg = f"{tags_filename} failed to download from {repo_id}"
            raise FileNotFoundError(msg) from e

    usecols = cast("UsecolsArgType", ("name", "category"))
    df: pd.DataFrame = pd.read_csv(csv_path, usecols=usecols)
    return ModelLabels(
        names=df["name"].tolist(),
        rating_indices=list(
            np.where(df["category"] == LabelCategory.RATING.value)[0].astype(int),
        ),
        general_indices=list(
            np.where(df["category"] == LabelCategory.GENERAL.value)[0].astype(int),
        ),
        character_indices=list(
            np.where(df["category"] == LabelCategory.CHARACTER.value)[0].astype(int),
        ),
    )


def get_model_repo_id(model_name: ModelName) -> str:
    """Get repository ID from model name.

    Args:
        model_name: Supported model identifier

    Returns:
        Hugging Face repository ID

    Raises:
        ValueError: If model name is unknown
    """
    try:
        return MODEL_REPO_MAP[model_name]
    except KeyError as exc:  # pragma: no cover - defensive runtime guard
        valid_models = ", ".join(MODEL_REPO_MAP.keys())
        msg = f"Unknown model name: {model_name}. Valid models: {valid_models}"
        raise ValueError(msg) from exc
