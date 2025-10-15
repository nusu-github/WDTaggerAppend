"""CLI tool for preparing and uploading datasets."""

from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi

from wd_tagger_append.dataset_utils import load_dataset_from_folder
from wd_tagger_append.labels import ModelName, get_model_repo_id, load_labels_from_hub

app = typer.Typer(help="WD Tagger dataset preparation tool")


@app.command()
def prepare(
    dataset_folder: Annotated[Path, typer.Argument(help="Path to the dataset folder")],
    repo_id: Annotated[
        str,
        typer.Argument(help="Hugging Face repository ID (e.g., username/dataset-name)"),
    ],
    model_name: Annotated[
        ModelName,
        typer.Option(help="Model name to use (convnext, eva02-large, swinv2, vit-large, vit)"),
    ] = "eva02-large",
    private: Annotated[bool, typer.Option(help="Make the dataset private")] = True,
    token: Annotated[str | None, typer.Option(help="Hugging Face API token")] = None,
) -> None:
    """Prepare dataset and upload to Hugging Face Hub."""
    # Get model repository ID
    model_repo_id = get_model_repo_id(model_name)
    typer.echo(f"Using model: {model_repo_id}")

    # Load dataset
    typer.echo("Loading dataset...")
    dataset = load_dataset_from_folder(str(dataset_folder))

    # Load pretrained labels
    typer.echo("Loading pretrained labels...")
    pretrained_labels = load_labels_from_hub(repo_id=model_repo_id)
    typer.echo(f"Pretrained labels: {pretrained_labels.num_labels}")

    # Auto-complete username if not present in repo_id
    api = HfApi(token=token)
    if "/" not in repo_id:
        typer.echo("Fetching username from Hugging Face API...")
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{repo_id}"
        typer.echo(f"Auto-completed repository: {repo_id}")

    # Create repository if it doesn't exist
    typer.echo(f"Creating repository: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        typer.echo(f"Repository created/verified: {repo_id}")
    except Exception as e:
        typer.echo(f"Error creating repository: {e}", err=True)
        raise typer.Exit(1) from e

    # Add metadata to dataset info
    dataset.info.description = f"Prepared dataset for WD Tagger training (based on {model_repo_id})"

    # Upload dataset to Hub
    typer.echo(f"Uploading dataset to {repo_id}...")
    try:
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token,
        )
        typer.echo(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        typer.echo(f"Error uploading dataset: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo("Done!")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
