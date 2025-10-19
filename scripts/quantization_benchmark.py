"""Quantization error measurement tool for WD Tagger models.

This script loads a dataset (local images or Hugging Face dataset) and
measures the quantization error when using 4-bit or 8-bit quantized models
compared to FP32 baseline.

Usage:
    # Local image directory
    uv run pytest tests/test.py --dataset-path /path/to/images

    # Hugging Face dataset
    uv run pytest tests/test.py --dataset-name user/dataset-name

    # Specific quantization method
    uv run pytest tests/test.py --dataset-path /path --quantization 8bit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, BitsAndBytesConfig

if TYPE_CHECKING:
    from collections.abc import Iterator


def _collect_images(dataset_path: Path) -> Iterator[Image.Image]:
    """Load images from local directory or Hugging Face dataset."""
    if not dataset_path.exists():
        msg = f"Dataset path not found: {dataset_path}"
        raise FileNotFoundError(msg)

    if dataset_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        image_files = [f for f in dataset_path.rglob("*") if f.suffix.lower() in image_extensions]

        if not image_files:
            msg = f"No images found in {dataset_path}"
            raise ValueError(msg)

        for image_file in sorted(image_files):
            try:
                img = Image.open(image_file).convert("RGB")
                yield img
            except Exception as e:
                print(f"Warning: Failed to load {image_file}: {e}")
    else:
        msg = f"Expected directory, got file: {dataset_path}"
        raise ValueError(msg)


def _measure_quantization_error(
    model_fp32: AutoModelForImageClassification,
    model_quantized: AutoModelForImageClassification,
    image_processor: AutoImageProcessor,
    images: list[Image.Image],
    threshold: float = 0.35,
) -> dict:
    """Measure quantization error across images."""
    all_probs_fp32 = []
    all_probs_quantized = []

    for img in images:
        inputs = image_processor(images=[img], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(model_fp32.device)

        with torch.no_grad():
            outputs_fp32 = model_fp32(pixel_values)
            logits_fp32 = outputs_fp32.logits
            probs_fp32 = torch.sigmoid(logits_fp32).cpu().numpy()

            outputs_quantized = model_quantized(pixel_values)
            logits_quantized = outputs_quantized.logits
            probs_quantized = torch.sigmoid(logits_quantized).cpu().numpy()

        all_probs_fp32.append(probs_fp32)
        all_probs_quantized.append(probs_quantized)

    all_probs_fp32 = np.vstack(all_probs_fp32)
    all_probs_quantized = np.vstack(all_probs_quantized)

    prob_diff = np.abs(all_probs_quantized - all_probs_fp32)

    results = {
        "num_images": len(images),
        "num_labels": all_probs_fp32.shape[1],
        "max_prob_diff": prob_diff.max(),
        "mean_prob_diff": prob_diff.mean(),
        "std_prob_diff": prob_diff.std(),
        "p95_prob_diff": np.percentile(prob_diff, 95),
        "p99_prob_diff": np.percentile(prob_diff, 99),
    }

    top_k = max(1, int(all_probs_fp32.shape[1] * 0.1))
    top_k_indices_fp32 = np.argsort(all_probs_fp32, axis=1)[:, -top_k:]
    top_k_indices_quantized = np.argsort(all_probs_quantized, axis=1)[:, -top_k:]
    per_row_match = [
        np.intersect1d(a, b).size / top_k
        for a, b in zip(top_k_indices_fp32, top_k_indices_quantized, strict=False)
    ]
    results["top_10_percent_match_rate"] = float(np.mean(per_row_match))

    tags_fp32 = (all_probs_fp32 > threshold).astype(int)
    tags_quantized = (all_probs_quantized > threshold).astype(int)
    exact_matches = (tags_fp32 == tags_quantized).all(axis=1)
    results["exact_match_rate"] = exact_matches.mean()

    hamming_distance = np.abs(tags_fp32 - tags_quantized).sum(axis=1)
    results["mean_hamming_distance"] = hamming_distance.mean()
    results["max_hamming_distance"] = hamming_distance.max()
    results["hamming_loss"] = hamming_distance.mean() / all_probs_fp32.shape[1]

    return results


def main() -> None:
    """Run quantization error measurement."""
    parser = argparse.ArgumentParser(
        description="Measure quantization error for WD Tagger models on a dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to local image directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SmilingWolf/wd-eva02-large-tagger-v3",
        help="Model ID from Hugging Face Hub",
    )
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit"],
        default="4bit",
        help="Quantization method to test",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Tag confidence threshold for predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )

    args = parser.parse_args()

    print(f"Loading images from {args.dataset_path}...")
    images = []
    for image_path in _collect_images(args.dataset_path):
        images.append(image_path)
        if len(images) >= args.max_images:
            break
    print(f"Loaded {len(images)} images")

    print(f"Loading image processor: {args.model}")
    image_processor = AutoImageProcessor.from_pretrained(args.model)

    print(f"Loading FP32 model: {args.model}")
    model_fp32 = AutoModelForImageClassification.from_pretrained(
        args.model,
        device_map=args.device,
    ).eval()

    print(f"Loading {args.quantization} quantized model: {args.model}")
    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["head"],
            llm_int8_has_fp16_weight=False,
        )

    model_quantized = AutoModelForImageClassification.from_pretrained(
        args.model,
        device_map=args.device,
        quantization_config=quantization_config,
    ).eval()

    print("\nMeasuring quantization error...")
    results = _measure_quantization_error(
        model_fp32=model_fp32,
        model_quantized=model_quantized,
        image_processor=image_processor,
        images=images,
        threshold=args.threshold,
    )

    print("\n" + "=" * 60)
    print(f"Quantization Error Report ({args.quantization})")
    print("=" * 60)
    print(f"Images processed: {results['num_images']}")
    print(f"Label dimensions: {results['num_labels']}")
    print()
    print("Probability Space Errors:")
    print(f"  Max difference: {results['max_prob_diff']:.6f}")
    print(f"  Mean difference: {results['mean_prob_diff']:.6f}")
    print(f"  Std deviation: {results['std_prob_diff']:.6f}")
    print(f"  95th percentile: {results['p95_prob_diff']:.6f}")
    print(f"  99th percentile: {results['p99_prob_diff']:.6f}")
    print()
    print("Tag Prediction Agreement:")
    print(f"  Top-10% match rate: {results['top_10_percent_match_rate']:.4f}")
    print(f"  Exact match rate: {results['exact_match_rate']:.4f}")
    print()
    print(f"Hamming Distance (tag threshold={args.threshold:.2f}):")
    print(f"  Mean distance: {results['mean_hamming_distance']:.2f} tags")
    print(f"  Max distance: {results['max_hamming_distance']:.0f} tags")
    print(f"  Hamming loss: {results['hamming_loss']:.6f} ({results['hamming_loss'] * 100:.4f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
