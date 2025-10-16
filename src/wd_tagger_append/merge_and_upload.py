"""CLI tool for merging a PEFT adapter and uploading to the Hugging Face Hub with custom code."""

import tempfile
from pathlib import Path
from typing import Annotated

import typer
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

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

    # Construct the target repository ID
    # The model's push_to_hub will auto-complete the username if needed
    hub_repo_id = repo_name if "/" in repo_name else repo_name
    typer.echo(f"Target repository: {hub_repo_id}")

    # Read adapter config to get the correct number of labels
    typer.echo(f"Loading adapter config from: {adapter_path}")
    adapter_config = AutoConfig.from_pretrained(adapter_path)

    num_labels = adapter_config.num_labels
    label2id = getattr(adapter_config, "label2id", None)
    id2label = getattr(adapter_config, "id2label", None)

    typer.echo(f"Loading base model: {base_model_repo_id} with {num_labels} labels")
    base_model = AutoModelForImageClassification.from_pretrained(
        base_model_repo_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

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

        # Use the high-level push_to_hub API from the model
        # This automatically creates the repo if needed and handles username resolution
        typer.echo(f"Pushing model to Hub: {hub_repo_id}")
        merged_model.push_to_hub(  # pyright: ignore[reportAttributeAccessIssue]
            repo_id=hub_repo_id,
            token=token,
            private=private,
            commit_message="Upload merged model with custom processing code",
        )

        # Also push the processor separately to ensure all files are uploaded
        processor.push_to_hub(
            repo_id=hub_repo_id,
            token=token,
            commit_message="Upload image processor configuration",
        )

        # For custom processing code, we still need upload_file
        # since it's not part of the standard model/processor files
        from huggingface_hub import HfApi

        from wd_tagger_append.constants import CUSTOM_PROCESSOR_FILENAME

        api = HfApi(token=token)
        custom_file_path = tmpdir_path / CUSTOM_PROCESSOR_FILENAME
        if custom_file_path.exists():
            typer.echo(f"Uploading {CUSTOM_PROCESSOR_FILENAME}...")
            api.upload_file(
                path_or_fileobj=str(custom_file_path),
                path_in_repo=CUSTOM_PROCESSOR_FILENAME,
                repo_id=hub_repo_id,
                token=token,
                repo_type="model",
                commit_message=f"Add {CUSTOM_PROCESSOR_FILENAME}",
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
