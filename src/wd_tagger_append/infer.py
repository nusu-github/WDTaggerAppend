"""Inference CLI built from composable services."""

import shutil
from collections.abc import Callable, Iterable, Iterator, Sequence
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
    MODEL_REPO_MAP,
    LabelCatalogLoader,
    LabelData,
    LocalPathInspector,
    ModelIdentifier,
    PrecisionParser,
    ProbabilityFormatter,
    QuantizationConfigFactory,
)

IMAGE_EXTENSIONS = frozenset(extension.lower() for extension in Image.registered_extensions())


class ConfigurationError(ValueError):
    """Raised when user provided options cannot be reconciled."""


@dataclass(frozen=True)
class AdapterSettings:
    name_or_path: str | None
    revision: str | None
    token: str | None


@dataclass(frozen=True)
class RuntimeOptions:
    quantization: bool
    precision: str | None
    device_map: str | None


@dataclass(frozen=True)
class InferenceThresholds:
    general: float
    character: float


@dataclass(frozen=True)
class DirectoryPlan:
    mode: Literal["text", "copy", "move", "none"]
    summary_path: Path | None
    category: Literal["rating", "character", "general"]
    destination: Path | None
    overwrite: bool
    rating_threshold: float


@dataclass(frozen=True)
class InferenceResult:
    image_path: Path
    caption: str
    tag_string: str
    ratings: dict[str, float]
    character: dict[str, float]
    general: dict[str, float]


def _raise_bad_parameter(message: str, cause: Exception | None = None) -> NoReturn:
    error = cast("Exception", BadParameter(message))
    if cause is not None:
        error.__cause__ = cause
    raise error  # pyright: ignore[reportGeneralTypeIssues]


class AdapterResolver:
    """Load adapter configuration details when provided."""

    def __init__(self, settings: AdapterSettings) -> None:
        self._settings = settings

    def resolve(self) -> PeftConfig | None:
        if self._settings.name_or_path is None:
            return None
        return PeftConfig.from_pretrained(
            self._settings.name_or_path,
            revision=self._settings.revision,
            token=self._settings.token,
        )


class LabelMetadataService:
    """Load label metadata while respecting adapter overrides."""

    def __init__(self, warning_callback: Callable[[str], None] | None = None) -> None:
        self._loader = LabelCatalogLoader(warning_callback=warning_callback)

    def load(
        self,
        base_identifier: str,
        *,
        base_revision: str | None,
        base_token: str | None,
        labels_path: Path | None,
        adapter_settings: AdapterSettings,
        fallback_repo: str | None,
    ) -> LabelData:
        catalog = self._loader.load(
            base_identifier,
            base_revision=base_revision,
            base_token=base_token,
            labels_path=labels_path,
            adapter=adapter_settings.name_or_path,
            adapter_revision=adapter_settings.revision,
            adapter_token=adapter_settings.token,
            fallback_repo=fallback_repo,
        )
        return LabelData.from_catalog(catalog)


class ModelLoader:
    """Construct pretrained models with optional adapters attached."""

    def __init__(self, options: RuntimeOptions) -> None:
        self._options = options

    def load(
        self,
        base_identifier: str,
        *,
        num_labels: int,
        token: str | None,
        revision: str | None,
        adapter_settings: AdapterSettings,
    ) -> PreTrainedModel:
        try:
            precision_dtype = PrecisionParser.parse(self._options.precision)
        except ValueError as exc:  # pragma: no cover - validated via CLI
            raise ConfigurationError(str(exc)) from exc

        quantization_config = (
            QuantizationConfigFactory.for_classifier() if self._options.quantization else None
        )

        base_kwargs: dict[str, Any] = {
            "num_labels": num_labels,
            "ignore_mismatched_sizes": True,
        }

        if quantization_config is not None:
            base_kwargs["quantization_config"] = quantization_config
        if precision_dtype is not None and quantization_config is None:
            base_kwargs["torch_dtype"] = precision_dtype
        if self._options.device_map is not None:
            base_kwargs["device_map"] = self._options.device_map
        if token is not None:
            base_kwargs["token"] = token
        if revision is not None and not LocalPathInspector.exists(base_identifier):
            base_kwargs["revision"] = revision

        model = TimmWrapperForImageClassification.from_pretrained(base_identifier, **base_kwargs)

        if adapter_settings.name_or_path is not None:
            adapter_kwargs: dict[str, Any] = {"is_trainable": False}
            if adapter_settings.token is not None:
                adapter_kwargs["token"] = adapter_settings.token
            if adapter_settings.revision is not None:
                adapter_kwargs["revision"] = adapter_settings.revision
            model = PeftModel.from_pretrained(
                model,
                adapter_settings.name_or_path,
                **adapter_kwargs,
            )

        return cast("PreTrainedModel", model.eval())


class ImageBatchBuilder:
    """Prepare batched pixel tensors for the inference engine."""

    def __init__(self, inference_pipeline: ImageClassificationPipeline) -> None:
        self._pipeline = inference_pipeline

    def build(self, image_paths: Sequence[Path]) -> torch.Tensor:
        pixel_values: list[torch.Tensor] = []
        for path in image_paths:
            with Image.open(path) as image:
                inputs = self._pipeline.preprocess(image)
            tensor = inputs.get("pixel_values")
            if tensor is None:
                msg = f"Pipeline preprocess did not return pixel_values for {path}"
                raise RuntimeError(msg)

            pixel_values.append(cast("torch.Tensor", tensor))

        return torch.cat(pixel_values, dim=0)


class LogitNormalizer:
    """Convert raw logits into detached tensors ready for sigmoid."""

    @staticmethod
    def normalize(raw_logits: Any) -> torch.Tensor:
        if isinstance(raw_logits, torch.Tensor):
            tensor = raw_logits
        elif isinstance(raw_logits, (list, tuple)):
            items = [
                item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                for item in raw_logits
            ]
            tensor = torch.stack(items)
        else:
            tensor = torch.as_tensor(raw_logits)

        tensor = tensor.detach()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to("cpu")


class InferenceEngine:
    """Run forward passes and format outputs."""

    def __init__(
        self,
        inference_pipeline: ImageClassificationPipeline,
        formatter: ProbabilityFormatter,
        thresholds: InferenceThresholds,
    ) -> None:
        self._pipeline = inference_pipeline
        self._batch_builder = ImageBatchBuilder(inference_pipeline)
        self._formatter = formatter
        self._thresholds = thresholds

    def predict(self, image_paths: Sequence[Path]) -> list[InferenceResult]:
        if not image_paths:
            return []

        batch = self._batch_builder.build(image_paths)

        with torch.inference_mode():
            outputs = self._pipeline._forward({"pixel_values": batch})

        logits = outputs.get("logits") if isinstance(outputs, dict) else None
        if logits is None:
            msg = "Model output did not include logits."
            raise RuntimeError(msg)

        probabilities = torch.sigmoid(LogitNormalizer.normalize(logits))

        results: list[InferenceResult] = []
        for path, probs in zip(image_paths, probabilities, strict=False):
            caption, tags, ratings, character, general = self._formatter.format(
                probs,
                general_threshold=self._thresholds.general,
                character_threshold=self._thresholds.character,
            )
            results.append(
                InferenceResult(
                    image_path=path,
                    caption=caption,
                    tag_string=tags,
                    ratings=ratings,
                    character=character,
                    general=general,
                ),
            )

        return results


class ResultPrinter:
    """Render inference results for single-image runs."""

    def __init__(self, thresholds: InferenceThresholds) -> None:
        self._thresholds = thresholds

    def display(self, result: InferenceResult) -> None:
        typer.echo("--------")
        typer.echo(f"Caption: {result.caption}")
        typer.echo("--------")
        typer.echo(f"Tags: {result.tag_string}")
        typer.echo("--------")
        typer.echo("Ratings:")
        for name, score in result.ratings.items():
            typer.echo(f"  {name}: {score:.3f}")
        typer.echo("--------")
        typer.echo(f"Character tags (threshold={self._thresholds.character}):")
        for name, score in result.character.items():
            typer.echo(f"  {name}: {score:.3f}")
        typer.echo("--------")
        typer.echo(f"General tags (threshold={self._thresholds.general}):")
        for name, score in result.general.items():
            typer.echo(f"  {name}: {score:.3f}")


class TagExtractor:
    """Select tag names for directory processing."""

    def __init__(self, rating_threshold: float) -> None:
        self._rating_threshold = rating_threshold

    def names_for(
        self,
        result: InferenceResult,
        category: Literal["rating", "character", "general"],
    ) -> list[str]:
        if category == "rating":
            return [
                name for name, score in result.ratings.items() if score >= self._rating_threshold
            ]
        if category == "character":
            return list(result.character.keys())
        return list(result.general.keys()) if category == "general" else []


class DirectorySummaryWriter:
    """Create CSV summaries for directory runs."""

    def __init__(self, extractor: TagExtractor) -> None:
        self._extractor = extractor

    def write(self, results: Sequence[InferenceResult], output_path: Path, base_dir: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []
        for result in results:
            try:
                relative_path = result.image_path.relative_to(base_dir)
            except ValueError:
                relative_path = result.image_path

            rows.append(
                {
                    "image": str(relative_path),
                    "ratings": self._format(self._extractor.names_for(result, "rating")),
                    "character": self._format(self._extractor.names_for(result, "character")),
                    "general": self._format(self._extractor.names_for(result, "general")),
                },
            )

        df = pd.DataFrame.from_records(rows, columns=["image", "ratings", "character", "general"])
        df.to_csv(output_path, index=False)

    @staticmethod
    def _format(names: Iterable[str]) -> str:
        collected = list(names)
        return ", ".join(collected) if collected else "none"


class FileOrganizer:
    """Copy or move images into tag-specific folders."""

    def __init__(
        self,
        extractor: TagExtractor,
        *,
        mode: Literal["copy", "move"],
        destination_root: Path,
        category: Literal["rating", "character", "general"],
        overwrite: bool,
    ) -> None:
        self._extractor = extractor
        self._mode = mode
        self._destination_root = destination_root
        self._category = category
        self._overwrite = overwrite

    def organise(self, results: Sequence[InferenceResult]) -> tuple[int, int]:
        self._destination_root.mkdir(parents=True, exist_ok=True)
        processed = 0
        skipped = 0

        for result in results:
            tag_names = self._extractor.names_for(
                result,
                cast("Literal['rating', 'character', 'general']", self._category),
            )
            folder_name = ", ".join(tag_names) if tag_names else "no_tag"
            safe_name = self._make_safe(folder_name)
            target_dir = self._destination_root / safe_name
            target_dir.mkdir(parents=True, exist_ok=True)

            destination = target_dir / result.image_path.name
            if self._skip_if_same_path(result.image_path, destination):
                skipped += 1
                continue

            if destination.exists():
                if not self._overwrite:
                    skipped += 1
                    typer.echo(
                        f"Skipping '{result.image_path}' -> '{destination}' (exists)."
                        " Use --directory-overwrite to replace.",
                        err=True,
                    )
                    continue
                if destination.is_file():
                    destination.unlink()

            if self._mode == "copy":
                shutil.copy2(result.image_path, destination)
            else:
                shutil.move(result.image_path, destination)
            processed += 1

        return processed, skipped

    @staticmethod
    def _make_safe(name: str) -> str:
        sanitized = sanitize_filename(
            name or "no_tag",
            replacement_text="_",
            platform="universal",
        ).strip()
        sanitized = sanitized.strip(".")
        return sanitized or "no_tag"

    @staticmethod
    def _skip_if_same_path(source: Path, destination: Path) -> bool:
        try:
            if source.resolve() == destination.resolve():
                typer.echo(
                    f"Skipping '{source}' because destination is identical.",
                    err=True,
                )
                return True
        except FileNotFoundError:
            return False
        return False


class ImageEnumerator:
    """Collect image paths from directories."""

    def __init__(self, extensions: Iterable[str]) -> None:
        self._extensions = {extension.lower() for extension in extensions}

    def enumerate(self, directory: Path) -> list[Path]:
        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in self._extensions
        )


class BatchPlanner:
    """Yield slices for batch processing."""

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def plan(self, paths: Sequence[Path]) -> Iterator[list[Path]]:
        for start in range(0, len(paths), self._batch_size):
            yield list(paths[start : start + self._batch_size])


class DirectoryRunner:
    """Handle directory inference flow."""

    def __init__(
        self,
        engine: InferenceEngine,
        batch_size: int,
        plan: DirectoryPlan,
    ) -> None:
        self._engine = engine
        self._plan = plan
        self._enumerator = ImageEnumerator(IMAGE_EXTENSIONS)
        self._batch_planner = BatchPlanner(batch_size)

    def run(self, directory: Path) -> None:
        image_paths = self._enumerator.enumerate(directory)
        if not image_paths:
            typer.echo(f"No images found in {directory}", err=True)
            return

        typer.echo(f"Found {len(image_paths)} image(s) under {directory}")
        combined: list[InferenceResult] = []
        total = len(image_paths)
        processed = 0

        for batch_paths in self._batch_planner.plan(image_paths):
            batch_end = processed + len(batch_paths)
            typer.echo(f"Processing images {processed + 1}-{batch_end} of {total}")
            combined.extend(self._engine.predict(batch_paths))
            processed = batch_end

        if not combined:
            typer.echo("No images were processed successfully.", err=True)
            return

        extractor = TagExtractor(self._plan.rating_threshold)
        if self._plan.mode == "text":
            summary_path = self._plan.summary_path or (directory / "wd_tagger_results.csv")
            DirectorySummaryWriter(extractor).write(combined, summary_path, directory)
            typer.echo(f"Wrote directory summary to {summary_path}")
            return

        if self._plan.mode in {"copy", "move"}:
            destination_root = self._plan.destination or (
                directory / f"sorted_by_{self._plan.category}"
            )
            organiser = FileOrganizer(
                extractor,
                mode=cast("Literal['copy', 'move']", self._plan.mode),
                destination_root=destination_root,
                category=self._plan.category,
                overwrite=self._plan.overwrite,
            )
            processed_count, skipped = organiser.organise(combined)
            action = "Copied" if self._plan.mode == "copy" else "Moved"
            typer.echo(f"{action} {processed_count} file(s) to {destination_root}")
            if skipped:
                typer.echo(f"Skipped {skipped} file(s).", err=True)
            return

        typer.echo(f"Processed {len(combined)} image(s); no directory output requested.")


app = typer.Typer(help="WD Tagger v3 inference")


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
    adapter_settings = AdapterSettings(adapter, adapter_revision, adapter_token_final)

    identifier = ModelIdentifier(model, base_repo)
    base_identifier = identifier.resolve()

    if base_repo is None and model not in MODEL_REPO_MAP:
        typer.echo(
            "Model key not found in MODEL_REPO_MAP. Treating value as repo ID or local path.",
            err=False,
        )

    adapter_config = AdapterResolver(adapter_settings).resolve()
    if adapter_config is not None:
        typer.echo(f"Adapter trained on base model '{adapter_config.base_model_name_or_path}'.")
        if adapter_config.base_model_name_or_path is not None:
            base_identifier = adapter_config.base_model_name_or_path
            typer.echo(f"Using '{base_identifier}' for inference.")

    typer.echo("Loading label metadata...")

    def _emit_label_message(message: str) -> None:
        typer.echo(message, err=message.startswith("Warning:"))

    label_service = LabelMetadataService(warning_callback=_emit_label_message)
    labels = label_service.load(
        base_identifier,
        base_revision=revision,
        base_token=token,
        labels_path=labels_path,
        adapter_settings=adapter_settings,
        fallback_repo=(
            adapter_config.base_model_name_or_path if adapter_config is not None else None
        ),
    )

    runtime_options = RuntimeOptions(
        quantization=quantization,
        precision=precision,
        device_map=device_map,
    )
    model_loader = ModelLoader(runtime_options)

    model_instance: PreTrainedModel
    try:
        model_instance = model_loader.load(
            base_identifier,
            num_labels=len(labels.names),
            token=token,
            revision=revision,
            adapter_settings=adapter_settings,
        )
    except ConfigurationError as exc:
        _raise_bad_parameter(str(exc), cause=exc)

    typer.echo("Creating data transform...")
    image_processor = WDTaggerImageProcessor(base_identifier)

    typer.echo("Setting up inference pipeline...")
    inference_pipeline = cast(
        "ImageClassificationPipeline",
        pipeline(
            task="image-classification",
            model=model_instance,
            image_processor=image_processor,
        ),
    )

    thresholds = InferenceThresholds(general=gen_threshold, character=char_threshold)
    formatter = ProbabilityFormatter(labels.to_catalog())
    engine = InferenceEngine(inference_pipeline, formatter, thresholds)

    if resolved_input.is_file():
        typer.echo("Running inference on single image...")
        results = engine.predict([resolved_input])
        if not results:
            typer.echo("Inference produced no results.", err=True)
            return
        ResultPrinter(thresholds).display(results[0])
        typer.echo("Done!")
        return

    plan = DirectoryPlan(
        mode=directory_output_mode,
        summary_path=directory_summary_path,
        category=directory_category,
        destination=directory_destination,
        overwrite=directory_overwrite,
        rating_threshold=rating_threshold,
    )
    DirectoryRunner(engine, batch_size, plan).run(resolved_input)
    typer.echo("Done!")


if __name__ == "__main__":
    app()
