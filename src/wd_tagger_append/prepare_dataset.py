"""Dataset preparation module for WD Tagger training.

This module provides utilities to convert image folders with Danbooru JSON metadata
into Hugging Face Datasets format using lightweight standard library processing.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from PIL import Image

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage, Sequence, Value

from .dataset_utils import RATING_CODE_TO_NAME

app = typer.Typer(help="Prepare WD Tagger datasets from image folders")

IMAGE_EXTENSIONS = {ext.lower() for ext in Image.registered_extensions()}


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


def _normalize_tag_field(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return []


def scan_and_parse_bulk(folder_path: Path) -> list[dict[str, Any]]:
    """Scan folder and parse all JSON files into a list of records."""
    if not folder_path.exists():
        msg = f"Folder not found: {folder_path}"
        raise ValueError(msg)

    typer.echo("Scanning directory and parsing JSON files...")

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
            typer.echo(f"Warning: Error parsing {json_path.name}: {exc}", err=True)
            continue

        md5 = data.get("md5")
        if not md5:
            typer.echo(f"Warning: Missing md5 in {json_path.name}", err=True)
            continue

        image_path = json_path.with_suffix("")
        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            missing_images += 1
            typer.echo(f"Warning: Missing image for {json_path.name}", err=True)
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
                "general": _normalize_tag_field(data.get("tag_string_general", [])),
                "character": _normalize_tag_field(data.get("tag_string_character", [])),
            },
        }

        records_by_md5[md5] = record

    if not records_by_md5:
        msg = f"No valid image+JSON pairs found in {folder_path}"
        raise ValueError(msg)

    typer.echo(f"Found {len(json_files)} metadata file(s)")
    if missing_images:
        typer.echo(f"Skipped {missing_images} file(s) due to missing images.", err=True)
    if duplicates:
        typer.echo(f"Removed {duplicates} duplicate images based on MD5")

    records = list(records_by_md5.values())
    typer.echo(f"Retained {len(records)} unique images")

    return records


def records_to_dataset(records: list[dict[str, Any]]) -> Dataset:
    """Convert parsed records to a Hugging Face Dataset with proper schema."""
    typer.echo("Converting records to Dataset...")
    return Dataset.from_list(records, features=get_dataset_features())


def create_dataset(image_dir: Path) -> Dataset:
    """Create a Hugging Face Dataset from an image directory using lightweight parsing."""
    records = scan_and_parse_bulk(image_dir)
    return records_to_dataset(records)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetDict:
    """Split dataset into train and validation sets."""
    typer.echo(f"Splitting dataset (train ratio: {train_ratio})...")

    split = dataset.train_test_split(test_size=1.0 - train_ratio, seed=seed)
    dataset_dict = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
        },
    )

    typer.echo(f"Train set: {len(dataset_dict['train'])} examples")
    typer.echo(f"Validation set: {len(dataset_dict['validation'])} examples")

    return dataset_dict


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

    dataset = create_dataset(image_dir)
    dataset_dict = split_dataset(dataset, train_ratio=train_ratio, seed=42)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Saving dataset to: {output_dir}")
        dataset_dict.save_to_disk(output_dir)
        typer.echo("Dataset saved successfully")

    if push_to_hub:
        if not hub_repo:
            typer.echo("Error: --hub-repo is required when --push-to-hub is set", err=True)
            raise typer.Exit(1)

        typer.echo(f"Pushing dataset to Hub: {hub_repo}")
        dataset_dict.push_to_hub(hub_repo, private=private)
        typer.echo("Dataset pushed to Hub successfully")

    typer.echo("Done!")


if __name__ == "__main__":
    app()
