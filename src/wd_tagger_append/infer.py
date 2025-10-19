import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast

import pandas as pd
import torch
import typer
from pathvalidate import sanitize_filename
from peft import PeftConfig, PeftModel
from PIL import Image
from transformers import PreTrainedModel, TimmWrapperForImageClassification, pipeline
from transformers.pipelines.image_classification import ImageClassificationPipeline
from typer import BadParameter

from .augmentation import WDTaggerImageProcessor
from .inference_utils import (
    LabelData,
    _create_quantization_config,
    _is_local_path,
    _load_label_data,
    _parse_precision,
    _resolve_base_model_identifier,
    get_tags,
)

IMAGE_EXTENSIONS = frozenset(extension.lower() for extension in Image.registered_extensions())


@dataclass
class InferenceResult:
    image_path: Path
    caption: str
    tag_string: str
    ratings: dict[str, float]
    character: dict[str, float]
    general: dict[str, float]


@dataclass(frozen=True)
class InferenceThresholds:
    general: float
    character: float


@dataclass(frozen=True)
class DirectoryOutputConfig:
    mode: Literal["text", "copy", "move", "none"]
    summary_path: Path | None
    category: Literal["rating", "character", "general"]
    destination: Path | None
    overwrite: bool
    rating_threshold: float


def _raise_bad_parameter(message: str, cause: Exception | None = None) -> NoReturn:
    error = cast("Exception", BadParameter(message))
    if cause is not None:
        error.__cause__ = cause
    raise error  # pyright: ignore[reportGeneralTypeIssues]


def _chunked_paths(paths: Sequence[Path], batch_size: int) -> Iterator[list[Path]]:
    for start in range(0, len(paths), batch_size):
        yield list(paths[start : start + batch_size])


def _normalize_logits(raw_logits: Any) -> torch.Tensor:
    if isinstance(raw_logits, torch.Tensor):
        tensor = raw_logits
    elif isinstance(raw_logits, (list, tuple)):
        tensor_list = [
            item if isinstance(item, torch.Tensor) else torch.as_tensor(item) for item in raw_logits
        ]
        tensor = torch.stack(tensor_list)
    else:
        tensor = torch.as_tensor(raw_logits)

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    tensor = tensor.detach()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to("cpu")


def _collect_directory_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
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
    if category == "general":
        return list(result.general.keys())
    return []


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


def _write_directory_summary(
    results: Sequence[InferenceResult],
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
    results: Sequence[InferenceResult],
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
            pass

        if destination.exists():
            if not overwrite:
                skipped += 1
                typer.echo(
                    f"Skipping '{result.image_path}' -> '{destination}' (exists). "
                    "Use --directory-overwrite to replace.",
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


class BatchInferenceRunner:
    def __init__(
        self,
        inference_pipeline: ImageClassificationPipeline,
        labels: LabelData,
        thresholds: InferenceThresholds,
    ) -> None:
        self._pipeline = inference_pipeline
        self._labels = labels
        self._thresholds = thresholds

    def predict(self, image_paths: Sequence[Path]) -> list[InferenceResult]:
        if not image_paths:
            return []

        inputs = [Image.open(path) for path in image_paths]
        model_inputs = self._pipeline.preprocess(inputs)

        with torch.inference_mode():
            outputs = self._pipeline._forward(model_inputs)

        logits = outputs.get("logits") if isinstance(outputs, dict) else None
        if logits is None:
            msg = "Model output did not include logits."
            raise RuntimeError(msg)

        normalized_logits = _normalize_logits(logits)
        probabilities = torch.sigmoid(normalized_logits)

        batch_results: list[InferenceResult] = []
        for path, probs in zip(image_paths, probabilities, strict=False):
            caption, taglist, ratings, character, general = get_tags(
                probs=probs,
                labels=self._labels,
                gen_threshold=self._thresholds.general,
                char_threshold=self._thresholds.character,
            )
            batch_results.append(
                InferenceResult(
                    image_path=path,
                    caption=caption,
                    tag_string=taglist,
                    ratings=ratings,
                    character=character,
                    general=general,
                ),
            )
        return batch_results


def _print_single_result(result: InferenceResult, thresholds: InferenceThresholds) -> None:
    typer.echo("--------")
    typer.echo(f"Caption: {result.caption}")
    typer.echo("--------")
    typer.echo(f"Tags: {result.tag_string}")
    typer.echo("--------")
    typer.echo("Ratings:")
    for name, score in result.ratings.items():
        typer.echo(f"  {name}: {score:.3f}")
    typer.echo("--------")
    typer.echo(f"Character tags (threshold={thresholds.character}):")
    for name, score in result.character.items():
        typer.echo(f"  {name}: {score:.3f}")
    typer.echo("--------")
    typer.echo(f"General tags (threshold={thresholds.general}):")
    for name, score in result.general.items():
        typer.echo(f"  {name}: {score:.3f}")


def _run_directory_inference(
    directory: Path,
    image_paths: Sequence[Path],
    runner: BatchInferenceRunner,
    batch_size: int,
    output_config: DirectoryOutputConfig,
) -> None:
    if not image_paths:
        typer.echo(f"No images found in {directory}", err=True)
        return

    typer.echo(f"Found {len(image_paths)} image(s) under {directory}")
    combined_results: list[InferenceResult] = []
    total = len(image_paths)
    processed = 0

    for batch_paths in _chunked_paths(image_paths, batch_size):
        batch_end = processed + len(batch_paths)
        typer.echo(f"Processing images {processed + 1}-{batch_end} of {total}")
        batch_results = runner.predict(batch_paths)
        combined_results.extend(batch_results)
        processed = batch_end

    if not combined_results:
        typer.echo("No images were processed successfully.", err=True)
        return

    if output_config.mode == "text":
        summary_path = output_config.summary_path or (directory / "wd_tagger_results.csv")
        _write_directory_summary(
            combined_results,
            output_path=summary_path,
            base_dir=directory,
            rating_threshold=output_config.rating_threshold,
        )
        typer.echo(f"Wrote directory summary to {summary_path}")
    elif output_config.mode in {"copy", "move"}:
        dest_root = output_config.destination or (directory / f"sorted_by_{output_config.category}")
        processed_count, skipped = _organize_files(
            combined_results,
            dest_root=dest_root,
            category=output_config.category,
            mode=cast("Literal['copy', 'move']", output_config.mode),
            rating_threshold=output_config.rating_threshold,
            overwrite=output_config.overwrite,
        )
        action = "Copied" if output_config.mode == "copy" else "Moved"
        typer.echo(f"{action} {processed_count} file(s) to {dest_root}")
        if skipped:
            typer.echo(f"Skipped {skipped} file(s).", err=True)
    else:
        typer.echo(f"Processed {len(combined_results)} image(s); no directory output requested.")


app = typer.Typer(help="WD Tagger v3 inference with timm")


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(help="Path to an image file or directory.")],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model key, repo ID, or local path for the base model."),
    ] = "eva02-large",
    base_repo: Annotated[
        str | None,
        typer.Option(
            "--base-repo",
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
        typer.Option("--general-threshold", help="Threshold for general tags."),
    ] = 0.35,
    char_threshold: Annotated[
        float,
        typer.Option("--character-threshold", help="Threshold for character tags."),
    ] = 0.75,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Number of images to process per forward pass when batching directories.",
        ),
    ] = 8,
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
    directory_output_mode: Annotated[
        Literal["text", "copy", "move", "none"],
        typer.Option(
            "--directory-output-mode",
            "--dir-output",
            help=(
                "Output mode when the input path is a directory. "
                "Choose from 'text', 'copy', 'move', or 'none'."
            ),
            case_sensitive=False,
        ),
    ] = "text",
    directory_summary_path: Annotated[
        Path | None,
        typer.Option(
            "--directory-summary-path",
            "--dir-text-file",
            help="Destination file for directory summaries when --directory-output-mode=text.",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    directory_category: Annotated[
        Literal["rating", "character", "general"],
        typer.Option(
            "--directory-category",
            "--dir-category",
            help=(
                "Tag category used when organizing directories with copy/move output. "
                "Ignored for text or none."
            ),
            case_sensitive=False,
        ),
    ] = "general",
    directory_destination: Annotated[
        Path | None,
        typer.Option(
            "--directory-destination",
            "--dir-destination",
            help=(
                "Root directory to create tag folders in when --directory-output-mode is copy or move. "
                "Defaults to '<input>/sorted_by_<category>'."
            ),
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    directory_overwrite: Annotated[
        bool,
        typer.Option(
            "--directory-overwrite/--no-directory-overwrite",
            "--dir-overwrite/--no-dir-overwrite",
            help="Overwrite files that already exist in the destination when copying or moving.",
        ),
    ] = False,
) -> None:
    resolved_input = input_path.expanduser().resolve()
    if not resolved_input.exists():
        _raise_bad_parameter(f"Path not found: {resolved_input}")
    if not (resolved_input.is_file() or resolved_input.is_dir()):
        _raise_bad_parameter(f"Path must be a file or directory: {resolved_input}")

    directory_output_mode = cast(
        "Literal['text', 'copy', 'move', 'none']",
        directory_output_mode.lower(),
    )
    directory_category = cast(
        "Literal['rating', 'character', 'general']",
        directory_category.lower(),
    )

    adapter_token_final = adapter_token or token
    base_identifier = _resolve_base_model_identifier(model, base_repo)

    peft_config: PeftConfig | None = None
    if adapter is not None:
        typer.echo(f"Loading adapter config from '{adapter}'...")
        peft_config = PeftConfig.from_pretrained(
            adapter,
            revision=adapter_revision,
            token=adapter_token_final,
        )
        adapter_base = peft_config.base_model_name_or_path
        if adapter_base is not None:
            base_identifier = adapter_base
        adapter_base_display = adapter_base or base_identifier
        typer.echo(
            "Adapter trained on base model "
            f"'{adapter_base_display}'. Using '{base_identifier}' for inference.",
        )

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
        _raise_bad_parameter(str(exc), cause=exc)

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
    hf_model_for_pipeline = cast("PreTrainedModel", hf_model)
    inference_pipeline = cast(
        "ImageClassificationPipeline",
        pipeline(
            task="image-classification",
            model=hf_model_for_pipeline,
            image_processor=image_processor,
        ),
    )

    thresholds = InferenceThresholds(
        general=gen_threshold,
        character=char_threshold,
    )
    runner = BatchInferenceRunner(
        inference_pipeline=inference_pipeline,
        labels=labels,
        thresholds=thresholds,
    )

    if resolved_input.is_file():
        typer.echo("Running inference on single image...")
        results = runner.predict([resolved_input])
        if not results:
            typer.echo("Inference produced no results.", err=True)
            return
        _print_single_result(results[0], thresholds)
        typer.echo("Done!")
        return

    image_paths = _collect_directory_images(resolved_input)
    output_config = DirectoryOutputConfig(
        mode=directory_output_mode,
        summary_path=directory_summary_path,
        category=directory_category,
        destination=directory_destination,
        overwrite=directory_overwrite,
        rating_threshold=rating_threshold,
    )
    _run_directory_inference(
        directory=resolved_input,
        image_paths=image_paths,
        runner=runner,
        batch_size=batch_size,
        output_config=output_config,
    )
    typer.echo("Done!")


if __name__ == "__main__":
    app()
