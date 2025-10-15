"""CLI tool for uploading tag CSV files to the Hugging Face Hub."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import typer
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from wd_tagger_append.dataset_utils import (  # noqa: F401
    DatasetSource,
    analyze_new_tags,
    detect_dataset_source,
    load_dataset_from_folder,
    load_dataset_from_hub,
    load_dataset_with_origin,
)
from wd_tagger_append.labels import (
    ModelName,
    get_model_repo_id,
    labels_to_dataframe,
    load_labels_from_hub,
)

DEFAULT_TAGS_FILENAME = "selected_tags.csv"

app = typer.Typer(help="Upload tag CSV files to the Hugging Face Hub")


@app.command()
def upload_tags(  # noqa: PLR0913 - CLI surface is intentional
    dataset_source: Annotated[
        str,
        typer.Argument(
            help="Dataset source for analysis (local folder or Hugging Face dataset repo ID)",
        ),
    ],
    repo_id: Annotated[
        str,
        typer.Argument(
            help="Target Hugging Face dataset repository ID (e.g., username/dataset-name)",
        ),
    ],
    model_name: Annotated[
        ModelName,
        typer.Option(help="Base model used for label initialisation"),
    ] = "eva02-large",
    repo_path: Annotated[
        str,
        typer.Option(
            "--repo-path",
            help="Destination path inside the repository",
        ),
    ] = DEFAULT_TAGS_FILENAME,
    output_csv: Annotated[
        Path | None,
        typer.Option(
            "--output-csv",
            help="Optional path to store the generated CSV locally",
        ),
    ] = None,
    commit_message: Annotated[
        str | None,
        typer.Option(help="Commit message to use for the upload"),
    ] = None,
    revision: Annotated[
        str | None,
        typer.Option(help="Branch or revision to push to"),
    ] = None,
    private: Annotated[
        bool,
        typer.Option(
            "--private/--public",
            help="Create or update the repository as private",
        ),
    ] = True,
    create_repo: Annotated[
        bool,
        typer.Option(
            "--create-repo/--no-create-repo",
            help="Create the repository if it does not exist",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(help="Generate the CSV without pushing to the Hub"),
    ] = False,
    token: Annotated[
        str | None,
        typer.Option(help="Hugging Face API token with write access"),
    ] = None,
) -> None:
    """Analyze a dataset, generate the tag CSV, and optionally upload it to the Hub."""
    dataset_origin = detect_dataset_source(dataset_source)
    if dataset_origin is DatasetSource.LOCAL:
        typer.echo(f"Loading dataset from local folder: {dataset_source}")
    else:
        typer.echo(f"Loading dataset from Hugging Face Hub: {dataset_source}")
    _, dataset = load_dataset_with_origin(
        dataset_source,
        token=token,
        source=dataset_origin,
    )

    model_repo_id = get_model_repo_id(model_name)
    typer.echo(f"Using base model labels from: {model_repo_id}")
    pretrained_labels = load_labels_from_hub(repo_id=model_repo_id, token=token)
    typer.echo(f"Base label count: {pretrained_labels.num_labels}")

    typer.echo("Analyzing dataset for new tags...")
    new_general_tags, new_character_tags = analyze_new_tags(dataset, pretrained_labels)
    typer.echo(f"  New general tags: {len(new_general_tags)}")
    typer.echo(f"  New character tags: {len(new_character_tags)}")

    extended_labels = pretrained_labels.extend_with_new_tags(
        new_general_tags,
        new_character_tags,
    )
    typer.echo(f"Total labels after extension: {extended_labels.num_labels}")

    dataframe = labels_to_dataframe(extended_labels)

    temp_path: Path | None = None
    if output_csv is not None:
        resolved_output = output_csv.expanduser().resolve()
    else:
        with NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp_file:
            resolved_output = Path(tmp_file.name)
        temp_path = resolved_output

    dataframe.to_csv(resolved_output, index=False)
    typer.echo(f"Tag CSV generated at: {resolved_output}")

    if dry_run:
        typer.echo("Dry run enabled; skipping upload.")
        if temp_path is not None:
            resolved_output.unlink(missing_ok=True)
        return

    normalized_repo_path = repo_path.lstrip("/")
    message = commit_message or f"Upload {resolved_output.name}"

    typer.echo(f"Uploading to {repo_id}:{normalized_repo_path}")
    api = HfApi(token=token)

    if create_repo:
        typer.echo(f"Ensuring repository exists: {repo_id}")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            typer.echo("Repository is ready.")
        except Exception as exc:  # pragma: no cover - defensive, unexpected
            typer.echo(f"Failed to create or verify repository: {exc}", err=True)
            if temp_path is not None:
                resolved_output.unlink(missing_ok=True)
            raise typer.Exit(1) from exc

    try:
        api.upload_file(
            path_or_fileobj=str(resolved_output),
            path_in_repo=normalized_repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=message,
            revision=revision,
        )
        revision_name = revision or "main"
        repo_url = (
            f"https://huggingface.co/datasets/{repo_id}/blob/{revision_name}/{normalized_repo_path}"
        )
        typer.echo(f"Upload completed: {repo_url}")
    except HfHubHTTPError as exc:
        typer.echo(f"Hugging Face Hub error: {exc}", err=True)
        if temp_path is not None:
            resolved_output.unlink(missing_ok=True)
        raise typer.Exit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive, unexpected
        typer.echo(f"Unexpected error during upload: {exc}", err=True)
        if temp_path is not None:
            resolved_output.unlink(missing_ok=True)
        raise typer.Exit(1) from exc
    else:
        if temp_path is not None:
            resolved_output.unlink(missing_ok=True)


def main() -> None:
    """Main entry point."""
    app()
