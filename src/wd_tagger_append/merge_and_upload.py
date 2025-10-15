"""CLI tool for merging a PEFT adapter and uploading to the Hugging Face Hub with custom code."""

import tempfile
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoImageProcessor, AutoModelForImageClassification

from wd_tagger_append.model_export import (
    configure_model_for_remote,
    copy_custom_processor_code,
)

app = typer.Typer(
    help=(
        "A tool to merge a PEFT adapter with its base model, "
        "add custom code, and upload to the Hub."
    ),
)


@app.command()
def merge_and_upload(
    base_model_repo_id: Annotated[
        str,
        typer.Argument(help="Base model repo ID from Hugging Face Hub"),
    ],
    adapter_path: Annotated[Path, typer.Argument(help="Path to the trained LoRA adapter")],
    repo_name: Annotated[
        str,
        typer.Argument(help="Repository name (without username, e.g., 'my-model')"),
    ],
    token: Annotated[str | None, typer.Option(help="Hugging Face API token")] = None,
    private: Annotated[bool, typer.Option(help="Make the repository private")] = True,
) -> None:
    """Merges a PEFT adapter, adds custom processing code, and uploads to the Hub."""
    if not adapter_path.is_dir():
        typer.echo(f"Error: Adapter path not found: {adapter_path}", err=True)
        raise typer.Exit(1)

    # Get username from API
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    hub_repo_id = f"{username}/{repo_name}"

    typer.echo(f"Target repository: {hub_repo_id}")

    typer.echo(f"Loading base model: {base_model_repo_id}")
    base_model = AutoModelForImageClassification.from_pretrained(base_model_repo_id)

    typer.echo(f"Loading PEFT adapter from: {adapter_path}")
    peft_model: PeftModel = PeftModel.from_pretrained(base_model, str(adapter_path))

    typer.echo("Merging adapter with base model...")
    merged_model: AutoModelForImageClassification = peft_model.merge_and_unload()  # pyright: ignore[reportCallIssue]
    typer.echo("Merge complete.")

    # Modify config for multi-label classification and custom processor
    configure_model_for_remote(merged_model)  # pyright: ignore[reportArgumentType]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        typer.echo(f"Saving merged model to temporary directory: {tmpdir}")
        merged_model.save_pretrained(tmpdir)  # pyright: ignore[reportAttributeAccessIssue]

        # Save original processor config
        processor = AutoImageProcessor.from_pretrained(base_model_repo_id)
        processor.save_pretrained(tmpdir)

        # Copy custom processor code
        custom_processor_dest = copy_custom_processor_code(tmpdir_path)
        typer.echo(f"Copied custom processor code to {custom_processor_dest}")

        typer.echo(f"Uploading contents of {tmpdir} to {hub_repo_id}...")
        api.create_repo(hub_repo_id, repo_type="model", exist_ok=True, private=private)
        api.upload_folder(
            repo_id=hub_repo_id,
            folder_path=tmpdir,
            repo_type="model",
        )

    typer.echo("Done!")
    typer.echo(f"Your model is now available at: https://huggingface.co/{hub_repo_id}")
    typer.echo("You can now load it with `trust_remote_code=True`:")
    typer.echo(
        f"  model = AutoModelForImageClassification.from_pretrained("
        f'"{hub_repo_id}", trust_remote_code=True)',
    )
    typer.echo(
        f"  processor = AutoImageProcessor.from_pretrained("
        f'"{hub_repo_id}", trust_remote_code=True)',
    )
    typer.echo("And use it in a pipeline:")
    typer.echo(
        f'  pipe = pipeline("image-classification", model="{hub_repo_id}", trust_remote_code=True)',
    )


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
