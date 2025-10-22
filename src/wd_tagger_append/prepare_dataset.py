"""Dataset preparation module for WD Tagger training.

This module provides utilities to convert image folders with Danbooru JSON metadata
into Hugging Face Datasets format using lightweight standard library processing.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from PIL import Image

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage, Sequence, Value

from .dataset_utils import RATING_CODE_TO_NAME, LabelMapping
from .stratification import (
    build_label_matrix,
    build_split_metadata,
    create_shuffle_splitter,
    dataset_from_indices,
    generate_shuffle_split_indices,
)

app = typer.Typer(help="Prepare WD Tagger datasets from image folders")

IMAGE_EXTENSIONS = {ext.lower() for ext in Image.registered_extensions()}


@dataclass(frozen=True)
class PreparationSettings:
    """Configuration for dataset splitting."""

    train_ratio: float
    seed: int = 42
    stratify_by_rating: bool = False
    min_samples_for_split: int = 2
    multilabel_stratification: bool = True


@dataclass(frozen=True)
class HubUploadSettings:
    """Capture Hub upload preferences."""

    enabled: bool
    repo: str | None
    private: bool


@dataclass(frozen=True)
class SplitResult:
    """Encapsulate split outputs alongside persisted metadata artifacts."""

    dataset_dict: DatasetDict
    metadata: dict[str, Any] | None = None
    label_mapping: LabelMapping | None = None


class TagNormalizer:
    """Normalize tag representations from Danbooru JSON metadata."""

    @staticmethod
    def normalize(value: Any) -> list[str]:
        if isinstance(value, str):
            return [item for item in value.split() if item]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        return []


class MetadataScanner:
    """Scan a directory and collect image metadata records."""

    def __init__(self, report: Callable[[str], None], warn: Callable[[str], None]) -> None:
        self._report = report
        self._warn = warn

    def scan(self, folder_path: Path) -> list[dict[str, Any]]:
        if not folder_path.exists():
            msg = f"Folder not found: {folder_path}"
            raise ValueError(msg)

        self._report("Scanning directory and parsing JSON files...")

        json_files = sorted(folder_path.rglob("*.json"))
        if not json_files:
            msg = f"No JSON metadata files found in {folder_path}"
            raise ValueError(msg)

        records_by_md5: dict[str, dict[str, Any]] = {}
        missing_images = 0
        duplicates = 0

        for json_path in json_files:
            try:
                with json_path.open() as handle:
                    data = json.load(handle)
            except json.JSONDecodeError as exc:
                self._warn(f"Warning: Error parsing {json_path.name}: {exc}")
                continue

            md5 = data.get("md5")
            if not md5:
                self._warn(f"Warning: Missing md5 in {json_path.name}")
                continue

            image_path = json_path.with_suffix("")
            if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                missing_images += 1
                self._warn(f"Warning: Missing image for {json_path.name}")
                continue

            if md5 in records_by_md5:
                duplicates += 1
                continue

            rating_code = data.get("rating", "g")
            rating = RATING_CODE_TO_NAME.get(rating_code, rating_code or "general")

            record = {
                "image": str(image_path),
                "md5": md5,
                "source": data.get("source", ""),
                "score": int(data.get("score", 0) or 0),
                "rating": rating,
                "tags": {
                    "general": TagNormalizer.normalize(data.get("tag_string_general", [])),
                    "character": TagNormalizer.normalize(data.get("tag_string_character", [])),
                },
            }

            records_by_md5[md5] = record

        if not records_by_md5:
            msg = f"No valid image+JSON pairs found in {folder_path}"
            raise ValueError(msg)

        self._report(f"Found {len(json_files)} metadata file(s)")
        if missing_images:
            self._warn(f"Skipped {missing_images} file(s) due to missing images.")
        if duplicates:
            self._report(f"Removed {duplicates} duplicate images based on MD5")

        records = list(records_by_md5.values())
        self._report(f"Retained {len(records)} unique images")

        return records


class DatasetFactory:
    """Create Hugging Face datasets from parsed records."""

    def __init__(self, report: Callable[[str], None]) -> None:
        self._report = report

    def build(self, records: list[dict[str, Any]]) -> Dataset:
        self._report("Converting records to Dataset...")
        return Dataset.from_list(records, features=get_dataset_features())


class SplitPlanner:
    """Split datasets into train and validation partitions."""

    def __init__(self, settings: PreparationSettings, report: Callable[[str], None]) -> None:
        self._settings = settings
        self._report = report

    def split(self, dataset: Dataset) -> SplitResult:
        dataset_size = len(dataset)

        if dataset_size < self._settings.min_samples_for_split:
            self._report(
                f"Warning: Dataset has only {dataset_size} sample(s). "
                "Skipping split and using all data for training.",
            )
            dataset_dict = DatasetDict({"train": dataset, "validation": dataset.select([])})
            return SplitResult(dataset_dict)

        desired_test_ratio = 1.0 - self._settings.train_ratio
        n_test = max(1, int(dataset_size * desired_test_ratio))
        n_train = dataset_size - n_test

        if n_train < 1:
            self._report(
                f"Warning: Train ratio {self._settings.train_ratio} would result in "
                f"empty training set (dataset size: {dataset_size}). "
                "Using all data for training.",
            )
            dataset_dict = DatasetDict({"train": dataset, "validation": dataset.select([])})
            return SplitResult(dataset_dict)

        self._report(
            f"Splitting dataset (train ratio: {self._settings.train_ratio})...",
        )

        effective_test_ratio = n_test / dataset_size
        split_kwargs = {
            "test_size": effective_test_ratio,
            "seed": self._settings.seed,
        }

        if self._settings.multilabel_stratification:
            self._report("Using multilabel stratified split...")
            try:
                label_matrix_result = build_label_matrix(dataset)
                splitter = create_shuffle_splitter(
                    test_size=effective_test_ratio,
                    seed=self._settings.seed,
                )
                train_indices, validation_indices = generate_shuffle_split_indices(
                    label_matrix_result.matrix,
                    splitter,
                )
                dataset_with_labels = dataset.add_column(
                    "label_matrix",
                    label_matrix_result.matrix.tolist(),
                )
                dataset_dict = dataset_from_indices(
                    dataset_with_labels,
                    train_indices,
                    validation_indices,
                )
                metadata = build_split_metadata(
                    splitter_name=splitter.__class__.__name__,
                    seed=self._settings.seed,
                    params={"test_size": effective_test_ratio, "n_splits": splitter.n_splits},
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    matrix_shape=label_matrix_result.matrix.shape,
                )
                self._report(f"Train set: {len(dataset_dict['train'])} examples")
                self._report(f"Validation set: {len(dataset_dict['validation'])} examples")
                return SplitResult(dataset_dict, metadata, label_matrix_result.mapping)
            except ValueError as exc:
                self._report(
                    f"Warning: Multilabel stratification failed ({exc}). "
                    "Falling back to rating or random split.",
                )

        if self._settings.stratify_by_rating:
            self._report("Using stratified split by rating...")
            try:
                split = dataset.train_test_split(
                    stratify_by_column="rating",
                    **split_kwargs,
                )
            except ValueError as exc:
                self._report(
                    f"Warning: Stratification failed ({exc}). Falling back to random split.",
                )
                split = dataset.train_test_split(**split_kwargs)
        else:
            split = dataset.train_test_split(**split_kwargs)

        dataset_dict = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
            },
        )

        self._report(f"Train set: {len(dataset_dict['train'])} examples")
        self._report(f"Validation set: {len(dataset_dict['validation'])} examples")

        return SplitResult(dataset_dict)


class DatasetSaver:
    """Persist datasets to disk or the Hugging Face Hub."""

    def __init__(self, report: Callable[[str], None], warn: Callable[[str], None]) -> None:
        self._report = report
        self._warn = warn

    def save_to_disk(self, dataset_dict: DatasetDict, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._report(f"Saving dataset to: {output_dir}")
        dataset_dict.save_to_disk(output_dir)
        self._report("Dataset saved successfully")

    def save_split_metadata(self, metadata: Mapping[str, Any], output_dir: Path) -> Path:
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        path = metadata_dir / "splits.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        self._report(f"Saved split metadata to: {path}")
        return path

    def save_label_mapping(self, label_mapping: LabelMapping, output_dir: Path) -> Path:
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        path = metadata_dir / "label_mapping.json"
        label_mapping.save_as_json(path)
        self._report(f"Saved label mapping to: {path}")
        return path

    def push_to_hub(self, dataset_dict: DatasetDict, settings: HubUploadSettings) -> None:
        if not settings.repo:
            self._warn("Error: --hub-repo is required when --push-to-hub is set")
            raise typer.Exit(1)

        self._report(f"Pushing dataset to Hub: {settings.repo}")
        dataset_dict.push_to_hub(settings.repo, private=settings.private)
        self._report("Dataset pushed to Hub successfully")


def get_dataset_features() -> Features:
    """Define the schema for WD Tagger datasets."""
    return Features(
        {
            "md5": Value("string"),
            "source": Value("string"),
            "image": HFImage(),
            "rating": ClassLabel(names=list(RATING_CODE_TO_NAME.values())),
            "tags": {
                "general": Sequence(Value("string")),
                "character": Sequence(Value("string")),
            },
            "score": Value("int32"),
        },
    )


@app.command()
def prepare(
    image_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing images and JSON metadata files",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save the processed dataset",
        ),
    ] = None,
    train_ratio: Annotated[
        float,
        typer.Option(
            "--train-ratio",
            "-r",
            min=0.0,
            max=1.0,
            help="Ratio of data to use for training",
        ),
    ] = 0.8,
    stratify_splits: Annotated[
        bool,
        typer.Option(
            "--stratify-splits/--no-stratify-splits",
            help="Use multilabel stratification when forming train/validation splits",
        ),
    ] = True,
    push_to_hub: Annotated[
        bool,
        typer.Option(
            "--push-to-hub",
            help="Upload dataset to Hugging Face Hub",
        ),
    ] = False,
    hub_repo: Annotated[
        str | None,
        typer.Option(
            "--hub-repo",
            help="Hugging Face Hub repository ID (e.g., 'username/dataset-name')",
        ),
    ] = None,
    private: Annotated[
        bool,
        typer.Option(
            "--private",
            help="Make the Hub repository private",
        ),
    ] = True,
) -> None:
    """Prepare WD Tagger dataset from image folder."""
    typer.echo(f"Preparing dataset from: {image_dir}")

    report = typer.echo

    def warn(message):
        return typer.echo(message, err=True)

    scanner = MetadataScanner(report, warn)
    factory = DatasetFactory(report)
    records = scanner.scan(image_dir)
    dataset = factory.build(records)

    preparation_settings = PreparationSettings(
        train_ratio=train_ratio,
        multilabel_stratification=stratify_splits,
    )
    split_planner = SplitPlanner(preparation_settings, report)
    split_result = split_planner.split(dataset)
    dataset_dict = split_result.dataset_dict

    saver = DatasetSaver(report, warn)

    if output_dir:
        saver.save_to_disk(dataset_dict, output_dir)
        if split_result.metadata is not None:
            saver.save_split_metadata(split_result.metadata, output_dir)
        if split_result.label_mapping is not None:
            saver.save_label_mapping(split_result.label_mapping, output_dir)

    hub_settings = HubUploadSettings(enabled=push_to_hub, repo=hub_repo, private=private)
    if hub_settings.enabled:
        saver.push_to_hub(dataset_dict, hub_settings)

    typer.echo("Done!")


if __name__ == "__main__":
    app()
