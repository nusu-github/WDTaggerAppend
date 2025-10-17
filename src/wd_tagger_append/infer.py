from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import pandas as pd
import torch
import typer
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from peft import PeftConfig, PeftModel
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from transformers import (
    AutoModelForImageClassification,
    BitsAndBytesConfig,
)

from wd_tagger_append.augmentation import create_eval_transform

app = typer.Typer(help="WD Tagger v3 inference with timm")

MODEL_REPO_MAP = {
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "vit-large": "SmilingWolf/wd-vit-large-tagger-v3",
    "eva02-large": "SmilingWolf/wd-eva02-large-tagger-v3",
}


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def _load_labels_from_csv(csv_path: Path) -> LabelData:
    """Load label metadata from a selected_tags.csv file."""
    df: pd.DataFrame = pd.read_csv(csv_path)
    return LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )


def _resolve_base_model_identifier(model: str, repo_override: str | None) -> str:
    """Resolve the base model repo or path from CLI inputs."""
    if repo_override:
        return repo_override
    if model in MODEL_REPO_MAP:
        return MODEL_REPO_MAP[model]
    return model


def _is_local_path(source: str) -> bool:
    """Return True if the given source refers to an existing local path."""
    try:
        return Path(source).expanduser().exists()
    except OSError:
        return False


def _create_quantization_config(
    load_in_4bit: bool,
    load_in_8bit: bool,
    compute_dtype: torch.dtype | None,
) -> BitsAndBytesConfig | None:
    """Construct a BitsAndBytes quantization configuration if requested."""
    if not load_in_4bit and not load_in_8bit:
        return None
    if load_in_4bit and load_in_8bit:
        msg = "Choose only one of --load-in-4bit or --load-in-8bit."
        raise typer.BadParameter(msg)

    kwargs: dict[str, Any] = {
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
    }
    if load_in_4bit:
        kwargs["bnb_4bit_use_double_quant"] = True
        kwargs["bnb_4bit_quant_type"] = "nf4"
        if compute_dtype is not None:
            kwargs["bnb_4bit_compute_dtype"] = compute_dtype
    return BitsAndBytesConfig(**kwargs)


def _parse_precision(precision: str | None) -> torch.dtype | None:
    """Parse precision flag into torch dtype."""
    if precision is None or precision.lower() == "fp32":
        return None
    normalized = precision.lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    msg = "Precision must be one of: fp32, bf16, fp16."
    raise typer.BadParameter(msg)


def _load_csv_from_source(
    source: str | None,
    adapter: str | None,
    adapter_revision: str | None,
    adapter_token: str | None,
) -> Path | None:
    """Locate a selected_tags.csv file from CLI options or adapter repo."""
    if source is not None:
        path = Path(source).expanduser()
        if not path.exists():
            msg = f"Label file not found: {source}"
            raise typer.BadParameter(msg)
        return path

    if adapter is None:
        return None

    # Try to resolve local adapter first.
    if _is_local_path(adapter):
        candidate = Path(adapter) / "selected_tags.csv"
        if candidate.exists():
            return candidate
        return None

    try:
        downloaded = hf_hub_download(
            repo_id=adapter,
            filename="selected_tags.csv",
            revision=adapter_revision,
            token=adapter_token,
        )
        return Path(downloaded)
    except HfHubHTTPError:
        return None


def _load_label_data(
    base_repo: str,
    revision: str | None,
    token: str | None,
    labels_path: Path | None,
    adapter: str | None,
    adapter_revision: str | None,
    adapter_token: str | None,
    fallback_repo: str | None = None,
) -> LabelData:
    """Load label metadata, falling back to the base model's selected_tags.csv."""
    base_labels: LabelData
    if _is_local_path(base_repo):
        local_csv = Path(base_repo).expanduser() / "selected_tags.csv"
        if local_csv.exists():
            base_labels = _load_labels_from_csv(local_csv)
        elif fallback_repo is not None and not _is_local_path(fallback_repo):
            typer.echo(
                f"selected_tags.csv not found locally; downloading labels from '{fallback_repo}'.",
            )
            base_labels = load_labels_hf(
                repo_id=fallback_repo,
                revision=revision,
                token=token,
            )
        else:
            typer.echo(
                "Warning: selected_tags.csv not found in local base model directory; "
                "category data for new labels will default to 'general'.",
                err=True,
            )
            base_labels = LabelData(names=[], rating=[], general=[], character=[])
    else:
        base_labels = load_labels_hf(repo_id=base_repo, revision=revision, token=token)

    csv_path = _load_csv_from_source(
        str(labels_path) if labels_path is not None else None,
        adapter=adapter,
        adapter_revision=adapter_revision,
        adapter_token=adapter_token,
    )
    if csv_path is None:
        return base_labels

    typer.echo(f"Loading labels from {csv_path}...")
    return _load_labels_from_csv(csv_path)


def load_labels_hf(
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
            token=token,
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        msg = f"selected_tags.csv failed to download from {repo_id}"
        raise FileNotFoundError(msg) from e

    return _load_labels_from_csv(csv_path)


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
) -> tuple[str, str, dict[str, float], dict[str, float], dict[str, float]]:
    # Convert model probabilities into python floats to keep typing simple
    prob_values = probs.detach().cpu().numpy().tolist()

    rating_labels: dict[str, float] = {
        labels.names[i]: float(prob_values[i]) for i in labels.rating
    }

    general_candidates = [
        (labels.names[i], float(prob_values[i]))
        for i in labels.general
        if prob_values[i] > gen_threshold
    ]
    gen_labels: dict[str, float] = dict(
        sorted(general_candidates, key=lambda item: item[1], reverse=True),
    )

    character_candidates = [
        (labels.names[i], float(prob_values[i]))
        for i in labels.character
        if prob_values[i] > char_threshold
    ]
    char_labels: dict[str, float] = dict(
        sorted(character_candidates, key=lambda item: item[1], reverse=True),
    )

    combined_names = [name for name, _ in general_candidates]
    combined_names.extend(name for name, _ in character_candidates)

    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", r"\(").replace(")", r"\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


@app.command()
def main(
    image_file: Annotated[Path, typer.Argument(help="Path to the image file")],
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
    load_in_4bit: Annotated[
        bool,
        typer.Option(
            "--load-in-4bit/--no-load-in-4bit",
            help="Enable 4-bit NF4 quantized loading (default: enabled).",
        ),
    ] = True,
    load_in_8bit: Annotated[
        bool,
        typer.Option(
            "--load-in-8bit/--no-load-in-8bit",
            help="Enable 8-bit quantized loading (default: disabled).",
        ),
    ] = False,
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
) -> None:
    image_path = image_file.resolve()
    if not image_path.is_file():
        typer.echo(f"Error: Image file not found: {image_path}", err=True)
        raise typer.Exit(code=1)

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
        adapter_base = cast(str, peft_config.base_model_name_or_path)
        if repo_id is None and model in MODEL_REPO_MAP:
            base_identifier = adapter_base
        typer.echo(
            "Adapter trained on base model "
            f"'{adapter_base}'. Using '{base_identifier}' for inference.",
        )

    # Load label metadata first to determine num_labels for model initialization
    typer.echo("Loading label metadata...")
    labels: LabelData = _load_label_data(
        base_repo=base_identifier,
        revision=revision,
        token=token,
        labels_path=labels_path,
        adapter=adapter,
        adapter_revision=adapter_revision,
        adapter_token=adapter_token_final,
        fallback_repo=(peft_config.base_model_name_or_path if peft_config is not None else None),
    )

    precision_dtype = _parse_precision(precision)
    quantization_config = _create_quantization_config(load_in_4bit, load_in_8bit, precision_dtype)

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
    hf_model = AutoModelForImageClassification.from_pretrained(
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
    transforms = create_eval_transform(base_identifier)

    typer.echo("Loading image and preprocessing...")
    img_input: Image.Image = Image.open(image_path)
    inputs = transforms(img_input).unsqueeze(0)  # Add batch dimension

    # Move inputs to the primary device if necessary.
    if hasattr(hf_model, "device"):
        inputs = inputs.to(hf_model.device)
    else:
        try:
            first_param = next(hf_model.parameters())
            inputs = inputs.to(first_param.device)
        except StopIteration:
            inputs = inputs.to("cpu")

    typer.echo("Running inference...")
    with torch.inference_mode():
        # run the model
        outputs = hf_model(inputs).logits
        # apply the final activation function (timm doesn't support doing this internally)
        outputs = F.sigmoid(outputs)

    typer.echo("Processing results...")
    caption, taglist, ratings, character, general = get_tags(
        probs=outputs.to("cpu").squeeze(0),
        labels=labels,
        gen_threshold=gen_threshold,
        char_threshold=char_threshold,
    )

    typer.echo("--------")
    typer.echo(f"Caption: {caption}")
    typer.echo("--------")
    typer.echo(f"Tags: {taglist}")

    typer.echo("--------")
    typer.echo("Ratings:")
    for k, v in ratings.items():
        typer.echo(f"  {k}: {v:.3f}")

    typer.echo("--------")
    typer.echo(f"Character tags (threshold={char_threshold}):")
    for k, v in character.items():
        typer.echo(f"  {k}: {v:.3f}")

    typer.echo("--------")
    typer.echo(f"General tags (threshold={gen_threshold}):")
    for k, v in general.items():
        typer.echo(f"  {k}: {v:.3f}")

    typer.echo("Done!")


if __name__ == "__main__":
    app()
