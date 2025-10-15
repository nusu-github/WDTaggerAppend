# WD Tagger Append

A comprehensive toolset for fine-tuning and deploying WD Tagger models for image tagging tasks. Built on top of the SmilingWolf WD Tagger family, this toolkit enables custom model training with LoRA adapters, dataset management, and streamlined inference.

## Features

- **Dataset Preparation**: Upload local datasets to Hugging Face Hub with automatic validation
- **Model Training**: Fine-tune WD Tagger models with LoRA adapters using local or Hub-hosted datasets
- **Inference**: Tag images using pre-trained or custom-trained models
- **Model Export**: Merge and upload trained adapters with custom processing code
- **Tag Management**: Generate and upload extended tag CSV files for custom label sets

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (recommended for training)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/nusu-github/WDTaggerAppend.git
cd WDTaggerAppend
uv sync
```

This will install all required dependencies including PyTorch with CUDA support on Linux/Windows, or CPU-only on other platforms.

The CLI commands can be invoked using the installed entry points (e.g., `wd-tagger-train`) or via `python -m` (e.g., `python -m wd_tagger_append.train`).

## Quick Start

### 1. Dataset Preparation and Upload

Prepare and upload a local dataset folder to Hugging Face Hub:

```bash
python -m wd_tagger_append.prepare_dataset datasets/your_dataset username/dataset-name \
  --model-name eva02-large \
  --private
```

**Options:**

- `--model-name`: Base model to use for label initialization (default: `eva02-large`)
  - Available models: `convnext`, `eva02-large`, `swinv2`, `vit-large`, `vit`
- `--private`: Create as a private dataset (default: `True`)
- `--token`: Hugging Face API token (reads from `HF_TOKEN` environment variable if not provided)

### 2. Model Training

Train a LoRA adapter on your dataset. The tool supports both local folders and Hugging Face Hub datasets.

#### Train from Local Dataset

```bash
python -m wd_tagger_append.train datasets/your_dataset \
  --model-name eva02-large \
  --output-dir ./output
```

#### Train from Hugging Face Hub Dataset

```bash
python -m wd_tagger_append.train username/dataset-name \
  --model-name eva02-large \
  --output-dir ./output \
  --token YOUR_TOKEN
```

**Training Options:**

- `--batch-size`: Training batch size per device (default: `4`)
- `--gradient-accumulation-steps`: Number of gradient accumulation steps (default: `4`)
- `--learning-rate`: Learning rate for AdamW optimizer (default: `5e-3`)
- `--num-epochs`: Total number of training epochs (default: `5`)
- `--lora-r`: LoRA attention dimension (default: `16`)
- `--lora-alpha`: LoRA scaling factor (default: `16`)
- `--lora-dropout`: LoRA dropout probability (default: `0.0`)
- `--test-size`: Validation split ratio (default: `0.1`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--fp16/--no-fp16`: Enable mixed precision training (default: `True` if CUDA available)

### 3. Inference

Run inference on images using pre-trained or custom-trained models:

```bash
python -m wd_tagger_append.infer path/to/image.jpg \
  --model-id-or-path SmilingWolf/wd-eva02-large-tagger-v3
```

**Using a Custom Trained Model:**

```bash
python -m wd_tagger_append.infer path/to/image.jpg \
  --model-id-or-path ./output/checkpoint-final \
  --gen-threshold 0.3 \
  --char-threshold 0.7
```

**Inference Options:**

- `--model-id-or-path`: Hugging Face Hub model ID or local path to model/adapter (default: `SmilingWolf/wd-eva02-large-tagger-v3`)
- `--gen-threshold`: Confidence threshold for general tags (default: `0.35`, range: `0.0-1.0`)
- `--char-threshold`: Confidence threshold for character tags (default: `0.75`, range: `0.0-1.0`)
- `--token`: Hugging Face API token for private models

### 4. Merge and Upload Model

After training, merge your LoRA adapter with the base model and upload to Hugging Face Hub:

```bash
python -m wd_tagger_append.merge_and_upload \
  SmilingWolf/wd-eva02-large-tagger-v3 \
  ./output/checkpoint-final \
  my-custom-tagger \
  --private \
  --token YOUR_TOKEN
```

This creates a standalone model with custom processing code that can be loaded with `trust_remote_code=True`.

### 5. Upload Extended Tags

Generate and upload an extended tag CSV file to your dataset repository:

```bash
python -m wd_tagger_append.upload_tags \
  datasets/your_dataset \
  username/dataset-name \
  --model-name eva02-large \
  --token YOUR_TOKEN
```

This analyzes your dataset for new tags not present in the base model and creates a `selected_tags.csv` file.

## Dataset Format

Organize your dataset with paired image and JSON metadata files:

```text
datasets/
  your_dataset/
    image1.jpg
    image1.jpg.json
    image2.png
    image2.png.json
    subfolder/
      image3.jpg
      image3.jpg.json
    ...
```

Each `.json` file must contain the following fields:

```json
{
  "md5": "abc123...",
  "rating": "g",
  "tags_general": ["1girl", "solo", "smile", "long_hair"],
  "tags_character": ["hatsune_miku"]
}
```

**Field Descriptions:**

- `md5`: MD5 hash of the image file (for verification)
- `rating`: Content rating - one of `general`, `sensitive`, `questionable`, or `explicit`
- `tags_general`: List of general descriptive tags
- `tags_character`: List of character name tags

## Complete Workflow Example

This example demonstrates the full pipeline from dataset preparation to model deployment.

### Step 1: Prepare Dataset (Machine A)

```bash
# Upload your local dataset to Hugging Face Hub
python -m wd_tagger_append.prepare_dataset datasets/anime_characters username/anime-characters-dataset \
  --model-name eva02-large \
  --private \
  --token $HF_TOKEN
```

### Step 2: Train Model (Machine B with GPU)

```bash
# Download from Hub and train with custom parameters
python -m wd_tagger_append.train username/anime-characters-dataset \
  --model-name eva02-large \
  --output-dir ./trained_adapters \
  --token $HF_TOKEN \
  --batch-size 8 \
  --num-epochs 10 \
  --learning-rate 1e-4 \
  --lora-r 32
```

### Step 3: Merge and Deploy

```bash
# Merge adapter with base model and upload
python -m wd_tagger_append.merge_and_upload \
  SmilingWolf/wd-eva02-large-tagger-v3 \
  ./trained_adapters/checkpoint-final \
  anime-character-tagger \
  --token $HF_TOKEN
```

### Step 4: Run Inference

```bash
# Use your deployed model for inference
python -m wd_tagger_append.infer test_image.jpg \
  --model-id-or-path username/anime-character-tagger \
  --gen-threshold 0.3 \
  --token $HF_TOKEN
```

## Development

### Code Quality

```bash
# Run linter
ruff check .

# Auto-format code
ruff format .

# Type checking
pyright
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_upload_tags.py

# Run with coverage
pytest --cov=wd_tagger_append
```

## Supported Models

This toolkit supports the following WD Tagger v3 models from SmilingWolf:

- `wd-convnext-tagger-v3`
- `wd-eva02-large-tagger-v3` (recommended)
- `wd-swinv2-tagger-v3`
- `wd-vit-large-tagger-v3`
- `wd-vit-tagger-v3`

## Acknowledgments

Built on top of the excellent WD Tagger models by [SmilingWolf](https://huggingface.co/SmilingWolf).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
