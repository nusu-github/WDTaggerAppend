import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
import torch
import typer
from pathvalidate import sanitize_filename
from peft import PeftConfig, PeftModel
from PIL import Image
from transformers import TimmWrapperForImageClassification, pipeline
from transformers.pipelines import ImageClassificationPipeline

from .augmentation import WDTaggerImageProcessor
from .inference_utils import (
    MODEL_REPO_MAP,
    LabelData,
    _create_quantization_config,
    _is_local_path,
    _load_label_data,
    _parse_precision,
    _resolve_base_model_identifier,
    get_tags,
)

IMAGE_EXTENSIONS = Image.registered_extensions().keys()


@dataclass
class InferenceResult:
    image_path: Path
    caption: str
    tag_string: str
    ratings: dict[str, float]
    character: dict[str, float]
    general: dict[str, float]


def _infer_single_image(
    image_path: Path,
    inference_pipeline: ImageClassificationPipeline,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
) -> InferenceResult:
    model_inputs = inference_pipeline.preprocess(str(image_path))

    with torch.inference_mode():
        model_outputs = inference_pipeline._forward(model_inputs)

    logits = model_outputs["logits"].to("cpu").squeeze(0)
    probs = torch.sigmoid(logits)

    caption, taglist, ratings, character, general = get_tags(
        probs=probs,
        labels=labels,
        gen_threshold=gen_threshold,
        char_threshold=char_threshold,
    )

    return InferenceResult(
        image_path=image_path,
        caption=caption,
        tag_string=taglist,
        ratings=ratings,
        character=character,
        general=general,
    )


def _gather_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _extract_tag_names(
    result: InferenceResult,
    category: Literal["rating", "character", "general"],
    rating_threshold: float,
) -> list[str]:
    if category == "rating":
        return [name for name, score in result.ratings.items() if score >= rating_threshold]
    if category == "character":
        return list(result.character.keys())
    return list(result.general.keys()) if category == "general" else []


def _format_tag_list(names: Iterable[str]) -> str:
    names_list = list(names)
    return ", ".join(names_list) if names_list else "none"


def _sanitize_folder_name(name: str) -> str:
    sanitized = sanitize_filename(
        name or "no_tag",
        replacement_text="_",
        platform="universal",
    ).strip()
    sanitized = sanitized.strip(".")
    return sanitized or "no_tag"


def _write_text_results(
    results: list[InferenceResult],
    output_path: Path,
    base_dir: Path,
    rating_threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for result in results:
        try:
            relative_path = result.image_path.relative_to(base_dir)
        except ValueError:
            relative_path = result.image_path

        records.append(
            {
                "image": str(relative_path),
                "ratings": _format_tag_list(
                    _extract_tag_names(result, "rating", rating_threshold),
                ),
                "character": _format_tag_list(
                    _extract_tag_names(result, "character", rating_threshold),
                ),
                "general": _format_tag_list(
                    _extract_tag_names(result, "general", rating_threshold),
                ),
            },
        )

    df = pd.DataFrame.from_records(records, columns=["image", "ratings", "character", "general"])
    df.to_csv(output_path, index=False)


def _organize_files(
    results: Iterable[InferenceResult],
    dest_root: Path,
    category: Literal["rating", "character", "general"],
    mode: Literal["copy", "move"],
    rating_threshold: float,
    overwrite: bool,
) -> tuple[int, int]:
    dest_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for result in results:
        tag_names = _extract_tag_names(result, category, rating_threshold)
        folder_name = ", ".join(tag_names) if tag_names else "no_tag"
        safe_folder_name = _sanitize_folder_name(folder_name)
        target_dir = dest_root / safe_folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        destination = target_dir / result.image_path.name
        try:
            if result.image_path.resolve() == destination.resolve():
                skipped += 1
                typer.echo(
                    f"Skipping '{result.image_path}' because destination is identical.",
                    err=True,
                )
                continue
        except FileNotFoundError:
            # One of the paths may not exist yet (e.g., destination). Ignore.
            pass

        if destination.exists():
            if not overwrite:
                skipped += 1
                typer.echo(
                    f"Skipping '{result.image_path}' -> '{destination}' (exists). "
                    "Use --dir-overwrite to replace.",
                    err=True,
                )
                continue
            if destination.is_file():
                destination.unlink()

        if mode == "copy":
            shutil.copy2(result.image_path, destination)
        else:
            shutil.move(result.image_path, destination)
        processed += 1

    return processed, skipped


def _print_single_result(
    result: InferenceResult,
    gen_threshold: float,
    char_threshold: float,
) -> None:
    typer.echo("--------")
    typer.echo(f"Caption: {result.caption}")
    typer.echo("--------")
    typer.echo(f"Tags: {result.tag_string}")

    typer.echo("--------")
    typer.echo("Ratings:")
    for name, score in result.ratings.items():
        typer.echo(f"  {name}: {score:.3f}")

    typer.echo("--------")
    typer.echo(f"Character tags (threshold={char_threshold}):")
    for name, score in result.character.items():
        typer.echo(f"  {name}: {score:.3f}")

    typer.echo("--------")
    typer.echo(f"General tags (threshold={gen_threshold}):")
    for name, score in result.general.items():
        typer.echo(f"  {name}: {score:.3f}")


def _process_directory(
    image_dir: Path,
    inference_pipeline: ImageClassificationPipeline,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
    dir_output: Literal["text", "copy", "move", "none"],
    dir_text_file: Path | None,
    dir_category: Literal["rating", "character", "general"],
    dir_destination: Path | None,
    dir_overwrite: bool,
    rating_threshold: float,
) -> None:
    image_paths = _gather_image_paths(image_dir)
    if not image_paths:
        typer.echo(f"No images found in {image_dir}", err=True)
        return

    typer.echo(f"Found {len(image_paths)} image(s) under {image_dir}")
    results: list[InferenceResult] = []
    for idx, path in enumerate(image_paths, start=1):
        typer.echo(f"[{idx}/{len(image_paths)}] Processing {path}")
        try:
            result = _infer_single_image(
                image_path=path,
                inference_pipeline=inference_pipeline,
                labels=labels,
                gen_threshold=gen_threshold,
                char_threshold=char_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Failed to process '{path}': {exc}", err=True)
            continue
        results.append(result)

    if not results:
        typer.echo("No images were processed successfully.", err=True)
        return

    if dir_output == "text":
        output_path = dir_text_file or (image_dir / "wd_tagger_results.txt")
        _write_text_results(
            results,
            output_path=output_path,
            base_dir=image_dir,
            rating_threshold=rating_threshold,
        )
        typer.echo(f"Wrote directory results to {output_path}")
    elif dir_output in {"copy", "move"}:
        dest_root = dir_destination or (image_dir / f"sorted_by_{dir_category}")
        mode = dir_output
        processed, skipped = _organize_files(
            results,
            dest_root=dest_root,
            category=dir_category,
            mode=mode,
            rating_threshold=rating_threshold,
            overwrite=dir_overwrite,
        )
        typer.echo(f"{mode.capitalize()}d {processed} file(s) to {dest_root}")
        if skipped:
            typer.echo(f"Skipped {skipped} file(s).", err=True)
    else:
        typer.echo(f"Processed {len(results)} image(s); no directory output requested.")


app = typer.Typer(help="WD Tagger v3 inference with timm")


@app.command()
def main(
    image_file: Annotated[Path, typer.Argument(help="Path to an image file or directory")],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model key, repo ID, or local path for the base model."),
    ] = "eva02-large",
    repo_id: Annotated[
        str | None,
        typer.Option(
            "--repo-id",
            help=(
                "Override the base model repo or directory (defaults to --model / MODEL_REPO_MAP)."
            ),
        ),
    ] = None,
    adapter: Annotated[
        str | None,
        typer.Option(
            "--adapter",
            help=("Optional PEFT LoRA adapter repo ID or local path produced by wd-tagger-train."),
        ),
    ] = None,
    labels_path: Annotated[
        Path | None,
        typer.Option(
            "--labels-path",
            help="Optional label_mapping.json or selected_tags.csv to override label metadata.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    revision: Annotated[
        str | None,
        typer.Option("--revision", help="Revision or branch to use for the base model."),
    ] = None,
    adapter_revision: Annotated[
        str | None,
        typer.Option("--adapter-revision", help="Revision or branch to use for the adapter."),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", help="Hugging Face token for private base models."),
    ] = None,
    adapter_token: Annotated[
        str | None,
        typer.Option(
            "--adapter-token",
            help="Hugging Face token for private adapters (defaults to --token).",
        ),
    ] = None,
    gen_threshold: Annotated[
        float,
        typer.Option(help="Threshold for general tags"),
    ] = 0.35,
    char_threshold: Annotated[
        float,
        typer.Option(help="Threshold for character tags"),
    ] = 0.75,
    quantization: Annotated[
        bool,
        typer.Option(
            "--quantization/--no-quantization",
            help="Enable 8-bit quantization for the model.",
        ),
    ] = True,
    precision: Annotated[
        str | None,
        typer.Option(
            "--precision",
            help="Precision for non-quantized weights: fp32, bf16, or fp16.",
        ),
    ] = None,
    device_map: Annotated[
        str | None,
        typer.Option("--device-map", help="Device map for model loading (default: auto)."),
    ] = "auto",
    rating_threshold: Annotated[
        float,
        typer.Option(
            "--rating-threshold",
            min=0.0,
            max=1.0,
            help=(
                "Minimum probability used to mark rating tags as detected when processing directories."
            ),
        ),
    ] = 0.5,
    dir_output: Annotated[
        Literal["text", "copy", "move", "none"],
        typer.Option(
            "--dir-output",
            help=(
                "Output mode when the input path is a directory. "
                "Choose from 'text', 'copy', 'move', or 'none'."
            ),
            case_sensitive=False,
        ),
    ] = "text",
    dir_text_file: Annotated[
        Path | None,
        typer.Option(
            "--dir-text-file",
            help="Destination file for directory summaries when --dir-output=text.",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    dir_category: Annotated[
        Literal["rating", "character", "general"],
        typer.Option(
            "--dir-category",
            help=(
                "Tag category used when organizing directories with copy/move output. "
                "Ignored for text or none."
            ),
            case_sensitive=False,
        ),
    ] = "general",
    dir_destination: Annotated[
        Path | None,
        typer.Option(
            "--dir-destination",
            help=(
                "Root directory to create tag folders in when --dir-output is copy or move. "
                "Defaults to '<input>/sorted_by_<category>'."
            ),
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    dir_overwrite: Annotated[
        bool,
        typer.Option(
            "--dir-overwrite/--no-dir-overwrite",
            help="Overwrite files that already exist in the destination when copying or moving.",
        ),
    ] = False,
) -> None:
    image_path = image_file.resolve()
    if not image_path.exists():
        typer.echo(f"Error: Path not found: {image_path}", err=True)
        raise typer.Exit(code=1)
    if not (image_path.is_file() or image_path.is_dir()):
        typer.echo(f"Error: Path must be a file or directory: {image_path}", err=True)
        raise typer.Exit(code=1)

    dir_output = dir_output.lower()
    dir_category = dir_category.lower()

    adapter_token_final = adapter_token or token
    base_identifier = _resolve_base_model_identifier(model, repo_id)

    peft_config: PeftConfig | None = None
    if adapter is not None:
        typer.echo(f"Loading adapter config from '{adapter}'...")
        peft_config = PeftConfig.from_pretrained(
            adapter,
            revision=adapter_revision,
            token=adapter_token_final,
        )
        adapter_base = peft_config.base_model_name_or_path
        if repo_id is None and model in MODEL_REPO_MAP:
            base_identifier = adapter_base
        typer.echo(
            "Adapter trained on base model "
            f"'{adapter_base}'. Using '{base_identifier}' for inference.",
        )

    # Load label metadata first to determine num_labels for model initialization
    typer.echo("Loading label metadata...")

    def _emit_label_message(message: str) -> None:
        typer.echo(message, err=message.startswith("Warning:"))

    labels: LabelData = _load_label_data(
        base_repo=base_identifier,
        revision=revision,
        token=token,
        labels_path=labels_path,
        adapter=adapter,
        adapter_revision=adapter_revision,
        adapter_token=adapter_token_final,
        fallback_repo=(peft_config.base_model_name_or_path if peft_config is not None else None),
        warning_callback=_emit_label_message,
    )

    try:
        precision_dtype = _parse_precision(precision)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    quantization_config = _create_quantization_config() if quantization else None

    base_kwargs: dict[str, Any] = {}
    if quantization_config is not None:
        base_kwargs["quantization_config"] = quantization_config
    if precision_dtype is not None and quantization_config is None:
        base_kwargs["torch_dtype"] = precision_dtype
    if device_map is not None:
        base_kwargs["device_map"] = device_map
    if token is not None:
        base_kwargs["token"] = token
    if revision is not None and not _is_local_path(base_identifier):
        base_kwargs["revision"] = revision

    # Specify num_labels to automatically expand classification head if needed
    num_labels = len(labels.names)
    base_kwargs["num_labels"] = num_labels
    base_kwargs["ignore_mismatched_sizes"] = True

    typer.echo(f"Loading base model from '{base_identifier}' with {num_labels} labels...")
    hf_model = TimmWrapperForImageClassification.from_pretrained(
        base_identifier,
        **base_kwargs,
    )

    if adapter is not None:
        adapter_kwargs: dict[str, Any] = {"is_trainable": False}
        if adapter_token_final is not None:
            adapter_kwargs["token"] = adapter_token_final
        if adapter_revision is not None:
            adapter_kwargs["revision"] = adapter_revision
        typer.echo(f"Applying adapter weights from '{adapter}'...")
        hf_model = PeftModel.from_pretrained(hf_model, adapter, **adapter_kwargs)

    hf_model = hf_model.eval()

    typer.echo("Creating data transform...")
    image_processor = WDTaggerImageProcessor(base_identifier)

    typer.echo("Setting up inference pipeline...")
    inference_pipeline = pipeline(
        task="image-classification",
        model=hf_model,
        image_processor=image_processor,
    )

    if image_path.is_dir():
        _process_directory(
            image_dir=image_path,
            inference_pipeline=inference_pipeline,
            labels=labels,
            gen_threshold=gen_threshold,
            char_threshold=char_threshold,
            dir_output=dir_output,
            dir_text_file=dir_text_file,
            dir_category=dir_category,
            dir_destination=dir_destination,
            dir_overwrite=dir_overwrite,
            rating_threshold=rating_threshold,
        )
        typer.echo("Done!")
        return

    typer.echo("Loading image and preprocessing...")
    result = _infer_single_image(
        image_path=image_path,
        inference_pipeline=inference_pipeline,
        labels=labels,
        gen_threshold=gen_threshold,
        char_threshold=char_threshold,
    )

    typer.echo("Running inference...")
    typer.echo("Processing results...")
    _print_single_result(
        result,
        gen_threshold=gen_threshold,
        char_threshold=char_threshold,
    )

    typer.echo("Done!")


if __name__ == "__main__":
    app()
