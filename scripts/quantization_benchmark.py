"""Quantization benchmark CLI for WD Tagger models.

This entry point compares a reference FP32 model against a BitsAndBytes quantized
variant across a sample of images drawn from a local directory or a Hugging Face
dataset. The script reuses the project image processor to ensure preprocessing
matches training and inference pipelines and reports agreement metrics built on
torchmetrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import torch
import typer
from PIL import Image
from torchmetrics import MeanAbsoluteError
from torchmetrics.classification import MultilabelExactMatch, MultilabelHammingDistance
from transformers import AutoModelForImageClassification, BitsAndBytesConfig

from datasets import Dataset, IterableDataset, load_dataset
from wd_tagger_append.augmentation import WDTaggerImageProcessor

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

app = typer.Typer(add_completion=False)

IMAGE_EXTENSIONS = {ext.lower() for ext in Image.registered_extensions()}


@dataclass
class BenchmarkInputs:
    processor: WDTaggerImageProcessor
    model_fp32: AutoModelForImageClassification
    model_quantized: AutoModelForImageClassification
    threshold: float
    topk_ratio: float
    batch_size: int
    max_samples: int | None


@dataclass
class BenchmarkResult:
    samples: int
    num_labels: int
    prob_mae: float
    prob_max: float
    prob_std: float
    prob_p95: float
    prob_p99: float
    topk_match_rate: float
    exact_match_rate: float
    hamming_distance: float


def _resolve_device(device_option: str) -> torch.device:
    if device_option == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_option)


def _resolve_device_map(device_option: str) -> str:
    return "auto" if device_option == "auto" else device_option


def _build_quantization_config(mode: Literal["4bit", "8bit"]) -> BitsAndBytesConfig:
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if mode == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["head"],
            llm_int8_has_fp16_weight=False,
        )
    msg = "Quantization mode must be either '4bit' or '8bit'."
    raise typer.BadParameter(msg)


def _batched(iterable: Iterable[Image.Image], batch_size: int) -> Iterator[list[Image.Image]]:
    iterator = iter(iterable)
    while True:
        if batch := list(islice(iterator, batch_size)):
            yield batch
        else:
            break


def _iter_local_images(root: Path) -> Iterator[Image.Image]:
    if not root.exists():
        msg = f"Dataset path not found: {root}"
        raise typer.BadParameter(msg)
    if not root.is_dir():
        msg = f"Expected directory, got file: {root}"
        raise typer.BadParameter(msg)

    image_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        msg = f"No images found in {root}"
        raise typer.BadParameter(msg)

    for path in image_files:
        try:
            with Image.open(path) as img:
                yield img.convert("RGB")
        except (OSError, ValueError) as exc:
            typer.echo(f"Skipping '{path}': {exc}", err=True)


def _iter_hf_dataset(
    dataset: Dataset | IterableDataset,
    image_column: str,
) -> Iterator[Image.Image]:
    for record in dataset:
        image_data = record[image_column]
        if isinstance(image_data, Image.Image):
            yield image_data.convert("RGB")
        else:
            yield Image.fromarray(image_data).convert("RGB")


def _collect_image_iterator(
    dataset_path: Path | None,
    dataset_name: str | None,
    dataset_config: str | None,
    dataset_split: str,
    image_column: str,
    streaming: bool,
) -> Iterator[Image.Image]:
    if dataset_path is not None and dataset_name is not None:
        msg = "Use either --dataset-path or --dataset-name, not both."
        raise typer.BadParameter(msg)
    if dataset_path is None and dataset_name is None:
        msg = "Provide --dataset-path for local files or --dataset-name for Hugging Face datasets."
        raise typer.BadParameter(msg)

    if dataset_path is not None:
        return _iter_local_images(dataset_path)

    dataset = load_dataset(
        path=dataset_name,
        name=dataset_config,
        split=dataset_split,
        streaming=streaming,
    )
    if image_column not in dataset.column_names:
        msg = f"Column '{image_column}' not present in dataset."
        raise typer.BadParameter(msg)
    return _iter_hf_dataset(dataset, image_column=image_column)


def _run_benchmark(
    image_iterator: Iterator[Image.Image],
    inputs: BenchmarkInputs,
    device: torch.device,
) -> BenchmarkResult:
    model_fp32 = inputs.model_fp32.to(device)
    model_fp32.eval()
    model_quantized = inputs.model_quantized
    model_quantized.eval()

    mae_metric = MeanAbsoluteError()
    hamming_metric: MultilabelHammingDistance | None = None
    exact_metric: MultilabelExactMatch | None = None

    prob_diffs: list[torch.Tensor] = []
    topk_match_sum = 0.0
    processed_samples = 0
    num_labels = None
    topk_ratio = inputs.topk_ratio
    batch_size = inputs.batch_size
    max_samples = inputs.max_samples

    def _take_limited_batches() -> Iterator[list[Image.Image]]:
        if max_samples is None:
            yield from _batched(image_iterator, batch_size)
            return

        remaining = max_samples
        for batch in _batched(image_iterator, batch_size):
            if remaining <= 0:
                break
            if len(batch) > remaining:
                yield batch[:remaining]
                break
            yield batch
            remaining -= len(batch)

    for batch in _take_limited_batches():
        if not batch:
            continue

        batch_inputs = inputs.processor(images=batch, return_tensors="pt")
        pixel_values = batch_inputs["pixel_values"].to(device)

        with torch.inference_mode():
            logits_fp32 = model_fp32(pixel_values).logits
            logits_quantized = model_quantized(pixel_values).logits

        probs_fp32 = torch.sigmoid(logits_fp32).detach().cpu()
        probs_quantized = torch.sigmoid(logits_quantized).detach().cpu()

        if num_labels is None:
            num_labels = probs_fp32.shape[1]
            hamming_metric = MultilabelHammingDistance(
                num_labels=num_labels,
                threshold=inputs.threshold,
            )
            exact_metric = MultilabelExactMatch(
                num_labels=num_labels,
                threshold=inputs.threshold,
            )

        mae_metric.update(probs_quantized, probs_fp32)
        hamming_metric.update(probs_quantized, probs_fp32)
        exact_metric.update(probs_quantized, probs_fp32)

        diff = (probs_quantized - probs_fp32).abs()
        prob_diffs.append(diff)

        k = max(1, int(topk_ratio * probs_fp32.shape[1]))
        topk_indices_fp32 = torch.topk(probs_fp32, k, dim=1).indices
        topk_indices_quantized = torch.topk(probs_quantized, k, dim=1).indices
        for indices_fp32, indices_quantized in zip(
            topk_indices_fp32,
            topk_indices_quantized,
            strict=False,
        ):
            overlap = torch.isin(indices_fp32, indices_quantized).sum().item()
            topk_match_sum += overlap / k

        processed_samples += probs_fp32.shape[0]

    if (
        processed_samples == 0
        or num_labels is None
        or hamming_metric is None
        or exact_metric is None
    ):
        msg = "No images were processed. Check dataset inputs."
        raise typer.BadParameter(msg)

    diff_tensor = torch.cat([tensor.flatten() for tensor in prob_diffs])
    prob_mae = float(mae_metric.compute())
    prob_max = float(diff_tensor.max())
    prob_std = float(diff_tensor.std(unbiased=False))
    prob_p95 = float(torch.quantile(diff_tensor, 0.95))
    prob_p99 = float(torch.quantile(diff_tensor, 0.99))

    return BenchmarkResult(
        samples=processed_samples,
        num_labels=num_labels,
        prob_mae=prob_mae,
        prob_max=prob_max,
        prob_std=prob_std,
        prob_p95=prob_p95,
        prob_p99=prob_p99,
        topk_match_rate=topk_match_sum / processed_samples,
        exact_match_rate=float(exact_metric.compute()),
        hamming_distance=float(hamming_metric.compute()),
    )


@app.command()
def main(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Model identifier, repository, or local path.",
        ),
    ] = "SmilingWolf/wd-eva02-large-tagger-v3",
    quantization: Annotated[
        Literal["4bit", "8bit"],
        typer.Option(
            "--quantization",
            case_sensitive=False,
            help="BitsAndBytes quantization mode to benchmark.",
        ),
    ] = "4bit",
    dataset_path: Annotated[
        Path | None,
        typer.Option(
            "--dataset-path",
            path_type=Path,
            help="Directory of images to evaluate.",
        ),
    ] = None,
    dataset_name: Annotated[
        str | None,
        typer.Option(
            "--dataset-name",
            help="Hugging Face dataset name to evaluate.",
        ),
    ] = None,
    dataset_config: Annotated[
        str | None,
        typer.Option(
            "--dataset-config",
            help="Optional dataset configuration.",
        ),
    ] = None,
    dataset_split: Annotated[
        str,
        typer.Option(
            "--dataset-split",
            help="Dataset split for Hugging Face datasets.",
        ),
    ] = "train",
    image_column: Annotated[
        str,
        typer.Option(
            "--image-column",
            help="Column containing images in the dataset.",
        ),
    ] = "image",
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            max=1.0,
            help="Threshold applied when comparing tag activations.",
        ),
    ] = 0.35,
    topk_ratio: Annotated[
        float,
        typer.Option(
            "--topk-ratio",
            min=0.0,
            max=1.0,
            help="Fraction of labels used for the top-k match rate.",
        ),
    ] = 0.1,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Batch size for inference.",
        ),
    ] = 4,
    max_samples: Annotated[
        int | None,
        typer.Option(
            "--max-samples",
            help="Maximum number of images to process. Defaults to all available samples.",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device for the FP32 model ('auto', 'cuda', 'cpu', ...).",
        ),
    ] = "auto",
    revision: Annotated[
        str | None,
        typer.Option(
            "--revision",
            help="Model revision or commit SHA.",
        ),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help="Authentication token for private models or datasets.",
        ),
    ] = None,
    streaming: Annotated[
        bool,
        typer.Option(
            "--streaming/--no-streaming",
            help="Enable streaming mode when loading Hugging Face datasets.",
        ),
    ] = False,
) -> None:
    if not 0.0 < topk_ratio <= 1.0:
        msg = "--topk-ratio must be between 0 and 1."
        raise typer.BadParameter(msg)

    device_resolved = _resolve_device(device)
    device_map = _resolve_device_map(device)

    typer.echo(f"Loading image processor for '{model}'...")
    processor = WDTaggerImageProcessor(
        pretrained_model_name_or_path=model,
        do_train_augmentations=False,
    )

    typer.echo("Loading FP32 reference model...")
    model_fp32 = AutoModelForImageClassification.from_pretrained(
        model,
        revision=revision,
        token=token,
        low_cpu_mem_usage=True,
    )

    typer.echo(f"Loading {quantization} quantized model...")
    quantization_mode = cast("Literal['4bit', '8bit']", quantization.lower())
    quantization_config = _build_quantization_config(quantization_mode)
    model_quantized = AutoModelForImageClassification.from_pretrained(
        model,
        revision=revision,
        token=token,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    image_iterator = _collect_image_iterator(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        image_column=image_column,
        streaming=streaming,
    )

    inputs = BenchmarkInputs(
        processor=processor,
        model_fp32=model_fp32,
        model_quantized=model_quantized,
        threshold=threshold,
        topk_ratio=topk_ratio,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    result = _run_benchmark(
        image_iterator=image_iterator,
        inputs=inputs,
        device=device_resolved,
    )

    typer.echo("\n================ Quantization Benchmark ================")
    typer.echo(f"Samples processed: {result.samples}")
    typer.echo(f"Number of labels: {result.num_labels}")
    typer.echo("\nProbability differences")
    typer.echo(f"  Mean absolute difference: {result.prob_mae:.6f}")
    typer.echo(f"  Maximum difference: {result.prob_max:.6f}")
    typer.echo(f"  Standard deviation: {result.prob_std:.6f}")
    typer.echo(f"  95th percentile: {result.prob_p95:.6f}")
    typer.echo(f"  99th percentile: {result.prob_p99:.6f}")
    typer.echo("\nClassification agreement")
    typer.echo(f"  Top-k match rate: {result.topk_match_rate:.4f}")
    typer.echo(f"  Exact match rate: {result.exact_match_rate:.4f}")
    typer.echo(f"  Mean Hamming distance: {result.hamming_distance:.4f}")
    typer.echo("========================================================")


if __name__ == "__main__":
    app()
