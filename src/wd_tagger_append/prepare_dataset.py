"""Dataset preparation module for WD Tagger training.

This module provides utilities to convert image folders with Danbooru JSON metadata
into Hugging Face Datasets format with MD5-based deduplication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from PIL import Image as PILImage

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value

if TYPE_CHECKING:
    from collections.abc import Generator

app = typer.Typer(help="Prepare WD Tagger datasets from image folders")


# Rating mapping from Danbooru single-letter codes to full names
RATING_MAP = {
    "g": "general",
    "s": "sensitive",
    "q": "questionable",
    "e": "explicit",
}


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
            "tags": {
                "rating": ClassLabel(names=list(RATING_MAP.values())),
                "general": Sequence(Value("string")),
                "character": Sequence(Value("string")),
            },
            "score": Value("int32"),
        },
    )


def scan_image_folder(folder_path: Path) -> list[tuple[Path, Path]]:
    """Scan folder for image + JSON pairs.

    Args:
        folder_path: Path to the folder containing images and JSON files.

    Returns:
        List of (image_path, json_path) tuples.

    Raises:
        ValueError: If folder doesn't exist or contains no valid pairs.
    """
    if not folder_path.exists():
        msg = f"Folder not found: {folder_path}"
        raise ValueError(msg)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    pairs = []

    for image_path in folder_path.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        json_path = image_path.parent / f"{image_path.name}.json"
        if json_path.exists():
            pairs.append((image_path, json_path))
        else:
            typer.echo(f"Warning: Missing JSON for {image_path.name}", err=True)

    if not pairs:
        msg = f"No valid image+JSON pairs found in {folder_path}"
        raise ValueError(msg)

    return pairs


def parse_danbooru_json(json_path: Path) -> dict:
    """Parse Danbooru JSON metadata.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Dictionary containing extracted metadata.

    Raises:
        ValueError: If JSON is malformed or missing required fields.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {json_path}: {e}"
        raise ValueError(msg) from e

    required_fields = ["md5", "rating", "score", "source"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        msg = f"Missing required fields in {json_path}: {', '.join(missing_fields)}"
        raise ValueError(msg)

    # Extract tag strings with fallback to empty string
    return {
        "md5": data["md5"],
        "score": data.get("score", 0),
        "source": data.get("source", ""),
        "tags": {
            "rating": data["rating"],
            "general": data.get("tag_string_general", "").split(),
            "character": data.get("tag_string_character", "").split(),
        },
    }


def create_dataset_generator(
    image_dir: Path,
) -> Generator[dict, None, None]:
    """Generate dataset examples with MD5-based deduplication.

    Args:
        image_dir: Path to the directory containing images and JSON files.

    Yields:
        Dictionary containing image data and metadata.
    """
    seen_md5 = set()
    pairs = scan_image_folder(image_dir)

    typer.echo(f"Found {len(pairs)} image+JSON pairs")

    for image_path, json_path in pairs:
        try:
            # Parse metadata
            metadata = parse_danbooru_json(json_path)
            md5_hash = metadata["md5"]

            # Check for duplicates
            if md5_hash in seen_md5:
                typer.echo(f"Skipping duplicate: {image_path.name} (md5: {md5_hash})")
                continue

            seen_md5.add(md5_hash)

            # Load image
            image = PILImage.open(image_path)

            # Convert rating
            rating = RATING_MAP.get(metadata["rating"], "general")

            yield {
                "image": image,
                "tags": metadata["tags"],
                "rating": rating,
                "score": metadata["score"],
                "source": metadata["source"],
                "md5": md5_hash,
            }

        except Exception as e:
            typer.echo(f"Error processing {image_path.name}: {e}", err=True)
            continue

    num_duplicates = len(pairs) - len(seen_md5)
    typer.echo(f"Processed {len(seen_md5)} unique images (skipped {num_duplicates} duplicates)")


def create_dataset(image_dir: Path) -> Dataset:
    """Create a Hugging Face Dataset from an image directory.

    Args:
        image_dir: Path to the directory containing images and JSON files.

    Returns:
        Dataset object containing the processed data.
    """
    typer.echo("Creating dataset from generator...")

    dataset = Dataset.from_generator(
        create_dataset_generator,
        gen_kwargs={"image_dir": image_dir},
        features=get_dataset_features(),
    )

    typer.echo(f"Dataset created with {len(dataset)} examples")
    return dataset


def split_dataset(
    dataset: Dataset,
    train_ratio: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
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
    """Prepare WD Tagger dataset from image folder with Danbooru JSON metadata.

    This command:
    1. Scans the image directory for image+JSON pairs
    2. Removes duplicates based on MD5 hash
    3. Splits data into train/validation sets
    4. Saves to disk in Arrow format (optional)
    5. Uploads to Hugging Face Hub (optional)
    """
    typer.echo(f"Preparing dataset from: {image_dir}")

    # Create dataset
    dataset = create_dataset(image_dir)

    # Split dataset
    dataset_dict = split_dataset(dataset, train_ratio=train_ratio)

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
