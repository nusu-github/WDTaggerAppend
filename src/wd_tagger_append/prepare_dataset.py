"""Dataset preparation module for WD Tagger training.

This module provides utilities to convert image folders with Danbooru JSON metadata
into Hugging Face Datasets format using Pandas for batch processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import PIL.Image
import typer

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value
from wd_tagger_append.dataset_utils import RATING_CODE_TO_NAME

app = typer.Typer(help="Prepare WD Tagger datasets from image folders")


def get_dataset_features() -> Features:
    """Define the schema for WD Tagger datasets.

    Returns:
        Features object defining the dataset schema.
    """
    return Features(
        {
            "md5": Value("string"),
            "source": Value("string"),
            "image": Image(),
            "rating": ClassLabel(names=list(RATING_CODE_TO_NAME.values())),
            "tags": {
                "general": Sequence(Value("string")),
                "character": Sequence(Value("string")),
            },
            "score": Value("int32"),
        },
    )


def scan_and_parse_bulk(folder_path: Path) -> pd.DataFrame:
    """Scan folder and parse all JSON files into a DataFrame in one pass.

    Uses Pandas operations for:
    - Vectorized data operations
    - MD5 deduplication via DataFrame.drop_duplicates()
    - Batch processing without Python-level iteration

    Args:
        folder_path: Path to the folder containing images and JSON files.

    Returns:
        DataFrame with columns: image_path, md5, rating, score, source, general_tags, character_tags

    Raises:
        ValueError: If folder doesn't exist or contains no valid pairs.
    """
    if not folder_path.exists():
        msg = f"Folder not found: {folder_path}"
        raise ValueError(msg)

    image_extensions = PIL.Image.registered_extensions().keys()
    records = []

    typer.echo("Scanning directory and parsing JSON files...")

    # Collect all image+JSON pairs
    for image_path in folder_path.rglob("**/*"):
        if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
            continue

        json_path = image_path.parent / f"{image_path.name}.json"
        if not json_path.exists():
            typer.echo(f"Warning: Missing JSON for {image_path.name}", err=True)
            continue

        # Parse JSON
        try:
            with open(json_path) as f:
                data = json.load(f)

            # Validate required fields
            if "md5" not in data:
                typer.echo(f"Warning: Missing md5 in {json_path.name}", err=True)
                continue

            records.append(
                {
                    "image_path": str(image_path),
                    "md5": data["md5"],
                    "rating": data.get("rating", "g"),
                    "score": data.get("score", 0),
                    "source": data.get("source", ""),
                    "general_tags": data.get("tag_string_general", "").split(),
                    "character_tags": data.get("tag_string_character", "").split(),
                },
            )

        except (json.JSONDecodeError, KeyError) as e:
            typer.echo(f"Warning: Error parsing {json_path.name}: {e}", err=True)
            continue

    if not records:
        msg = f"No valid image+JSON pairs found in {folder_path}"
        raise ValueError(msg)

    # Create DataFrame from all records in a single operation
    df = pd.DataFrame(records)
    typer.echo(f"Found {len(df)} image+JSON pairs")

    # Vectorized MD5 deduplication - native Pandas operation (C code)
    original_len = len(df)
    df = df.drop_duplicates(subset=["md5"], keep="first")
    num_duplicates = original_len - len(df)

    if num_duplicates > 0:
        typer.echo(f"Removed {num_duplicates} duplicate images based on MD5")

    typer.echo(f"Retained {len(df)} unique images")

    return df


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    """Convert DataFrame to Hugging Face Dataset with proper schema.

    Args:
        df: DataFrame with image metadata.

    Returns:
        Dataset object with proper features.
    """
    typer.echo("Converting DataFrame to Dataset...")

    # Prepare data in the format expected by Dataset
    data_dict = {
        "image": df["image_path"].tolist(),
        "md5": df["md5"].tolist(),
        "source": df["source"].tolist(),
        "score": df["score"].astype("int32").tolist(),
        "rating": [RATING_CODE_TO_NAME.get(r, "general") for r in df["rating"]],
        "tags": [
            {"general": gen, "character": char}
            for gen, char in zip(df["general_tags"], df["character_tags"], strict=True)
        ],
    }

    # Create dataset from dict with all data in memory
    dataset = Dataset.from_dict(data_dict, features=get_dataset_features())

    typer.echo(f"Dataset created with {len(dataset)} examples")
    return dataset


def create_dataset(image_dir: Path) -> Dataset:
    """Create a Hugging Face Dataset from an image directory using Pandas pipeline.

    Processing steps:
    - Scans and parses all JSON files in one pass
    - Deduplicates by MD5 using Pandas DataFrame operations
    - Converts to Dataset using batch operations

    Args:
        image_dir: Path to the directory containing images and JSON files.

    Returns:
        Dataset object containing the processed data.
    """
    # Step 1: Scan and parse everything into a DataFrame
    df = scan_and_parse_bulk(image_dir)

    # Step 2: Convert to Dataset
    return dataframe_to_dataset(df)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetDict:
    """Split dataset into train and validation sets.

    Args:
        dataset: Dataset to split.
        train_ratio: Ratio of data to use for training (default: 0.8).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
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
    """Prepare WD Tagger dataset from image folder.

    This command uses Pandas for vectorized operations:
    - Bulk JSON parsing and DataFrame creation
    - MD5 deduplication via Pandas
    - Single Dataset.from_dict() call

    Processing steps:
    1. Scans the image directory for image+JSON pairs
    2. Removes duplicates based on MD5 hash
    3. Splits data into train/validation sets
    4. Saves to disk in Arrow format (optional)
    5. Uploads to Hugging Face Hub (optional)
    """
    typer.echo(f"Preparing dataset from: {image_dir}")

    # Create dataset using Pandas pipeline
    dataset = create_dataset(image_dir)

    # Split dataset
    dataset_dict = split_dataset(dataset, train_ratio=train_ratio, seed=42)

    # Save to disk if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Saving dataset to: {output_dir}")
        dataset_dict.save_to_disk(output_dir)
        typer.echo("Dataset saved successfully")

    # Push to Hub if requested
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
