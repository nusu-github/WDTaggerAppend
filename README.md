# WD Tagger Append

Tools for extending the SmilingWolf WD Tagger models with custom datasets, LoRA fine-tuning, and streamlined inference. The project embraces the Hugging Face ecosystem (datasets, transformers, peft, accelerate, bitsandbytes) to keep the training loop lightweight while matching the tagger family’s preprocessing and label handling.

## Features

- **Dataset preparation** – Scan Danbooru-style folders, deduplicate by MD5, split into train/validation, optionally save locally and push to the Hugging Face Hub.
- **LoRA fine-tuning** – Train adapters for WD Tagger backbones (SwinV2, ConvNeXt, ViT, EVA02) with automatic classifier-head expansion and optional k-bit quantization.
- **Inference** – Tag images locally with threshold controls using WD Tagger backbones (extendable to custom adapters).

## Prerequisites

- Python 3.11 or newer
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU recommended for training (CPU and 4/8-bit modes are supported for experimentation)

## Installation

```bash
git clone https://github.com/nusu-github/WDTaggerAppend.git
cd WDTaggerAppend
uv sync
```

`uv sync` installs the managed virtual environment declared in `pyproject.toml`, including the correct PyTorch build (CUDA on Linux/Windows, CPU elsewhere).

Invoke the CLIs with `uv run wd-tagger-<command> …` or `python -m wd_tagger_append.<module> …`.

## Quick Start

### 1. Prepare a dataset

```bash
uv run wd-tagger-prepare datasets/anime_set \
  --output-dir datasets/processed/anime_set \
  --push-to-hub \
  --hub-repo username/anime-set \
  --private
```

Key options:

- `--output-dir`: Save the Arrow dataset to disk (omit to skip).
- `--push-to-hub`: Upload the resulting `DatasetDict`; needs `--hub-repo`.
- `--hub-repo`: Target repo such as `username/dataset-name`.
- `--private`: Create the Hub repo as private.

### 2. Train a LoRA adapter

```bash
uv run wd-tagger-train \
  --dataset-path datasets/processed/anime_set \
  --model eva02-large \
  --output-dir output/anime_set_eva02_lora \
  --train-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 3e-5 \
  --num-epochs 3 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --precision bf16
```

Alternative: `--dataset-name username/anime-set` (optionally with `--dataset-config`) to stream from the Hub.

Important knobs:

- `--model`: Backbone key from `MODEL_REPO_MAP` (`swinv2`, `convnext`, `vit`, `vit-large`, `eva02-large`).
- `--train-batch-size`, `--eval-batch-size`, `--gradient-accumulation-steps`: Control memory vs. throughput.
- `--learning-rate`, `--num-epochs`, `--weight-decay`, `--warmup-ratio`: Optimisation parameters.
- `--lora-rank`, `--lora-alpha`, `--lora-dropout`: PEFT LoRA hyperparameters.
- `--mixup-alpha`: Enable torchvision MixUp during batching.
- `--load-in-4bit` / `--load-in-8bit`: Activate BitsAndBytes quantised loading; works with `--precision` (`fp32`, `bf16`, `fp16`).
- `--metrics-top-k`: Report Precision@K / Recall@K / nDCG@K for chosen cut-offs (repeat to add values).
- `--metrics-propensity`: Add propensity-scored nDCG@K using dataset tag frequencies (requires `--metrics-top-k`).
- `--push-to-hub`, `--hub-model-id`, `--hub-token`: Push adapters automatically after training.

The trainer writes checkpoints plus a `label_mapping.json` (merged base + dataset tags) to `output-dir`.

### 3. Run inference

```bash
uv run wd-tagger-infer path/to/sample.jpg \
  --model eva02-large \
  --gen-threshold 0.35 \
  --char-threshold 0.7
```

Use a fine-tuned adapter (the CLI auto-detects `label_mapping.json` if it exists in the adapter directory):

```bash
uv run wd-tagger-infer path/to/sample.jpg \
  --model eva02-large \
  --adapter output/anime_set_eva02_lora \
  --gen-threshold 0.35 \
  --char-threshold 0.7
```

Options:

- `--model`: Backbone key, repo ID, or local base model path.
- `--repo-id`: Override the base model repo/path when `--model` is only a key.
- `--adapter`: Load a PEFT adapter from a local directory or Hub repo (e.g. `output/...`).
- `--labels-path`: Provide a custom `label_mapping.json` or `selected_tags.csv` (defaults to adapter/base assets).
- `--gen-threshold`, `--char-threshold`: Filter probabilities for general/character tags.
- `--load-in-4bit/--no-load-in-4bit`: Toggle 4-bit NF4 inference (default: on).
- `--load-in-8bit/--no-load-in-8bit`: Toggle 8-bit inference.
- `--token`: Hugging Face token for private repos.

## Dataset format

`wd-tagger-prepare` expects Danbooru-style `{image}.{ext}` + `{image}.{ext}.json` pairs. Minimal JSON structure:

```json
{
  "md5": "abc123…",
  "rating": "s",
  "score": 42,
  "source": "https://danbooru.donmai.us/posts/123",
  "tag_string_general": "1girl solo smile long_hair",
  "tag_string_character": "hatsune_miku",
  "tag_string_copyright": "vocaloid",
  "tag_string_artist": "piapro",
  "tag_string_meta": ""
}
```

Missing tag strings default to empty lists. Ratings follow Danbooru’s codes (`s`, `q`, `e`, etc.). The tool deduplicates by `md5`, records the `score`/`source`, and produces multi-hot labels for general, character, copyright, artist, and meta categories.

## Example pipeline

1. **Prepare** (machine A):

   ```bash
   uv run wd-tagger-prepare datasets/anime_characters \
     --push-to-hub \
     --hub-repo username/anime-characters \
     --private
   ```

2. **Train** (GPU machine B):

   ```bash
   uv run wd-tagger-train \
     --dataset-name username/anime-characters \
     --model eva02-large \
     --output-dir output/anime_characters_eva02 \
     --train-batch-size 8 \
     --gradient-accumulation-steps 2 \
     --learning-rate 3e-5 \
     --num-epochs 3 \
     --lora-rank 16 \
     --lora-alpha 32 \
     --precision bf16 \
     --push-to-hub \
     --hub-model-id username/anime-characters-eva02-lora
   ```

3. **Infer** (any machine):

   ```bash
   uv run wd-tagger-infer path/to/test.jpg \
     --model eva02-large \
     --adapter output/anime_characters_eva02 \
     --gen-threshold 0.4 \
     --char-threshold 0.6
   ```

## Development

```bash
# Formatting / linting
uv run ruff format .
uv run ruff check .

# Type checking
uv run pyright

# Tests (add selectors as needed)
uv run pytest
```

## Supported backbones

- `SmilingWolf/wd-swinv2-tagger-v3`
- `SmilingWolf/wd-convnext-tagger-v3`
- `SmilingWolf/wd-vit-tagger-v3`
- `SmilingWolf/wd-vit-large-tagger-v3`
- `SmilingWolf/wd-eva02-large-tagger-v3`

## License

Dual licensed under the MIT License and the Apache License 2.0. See `LICENSE-MIT` and `LICENSE-APACHE` for details.
