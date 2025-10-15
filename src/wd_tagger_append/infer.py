"""CLI tool for model inference."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import torch
import typer
from PIL import Image
from torch import Tensor
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TimmWrapperForImageClassification,
    TimmWrapperImageProcessor,
)

from wd_tagger_append.labels import ModelLabels, load_labels_from_hub

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers.modeling_outputs import ImageClassifierOutput

app = typer.Typer(help="WD Tagger image inference tool")

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tags(
    probs: Tensor,
    labels: ModelLabels,
    gen_threshold: float,
    char_threshold: float,
) -> tuple[str, str, dict[str, float], dict[str, float], dict[str, float]]:
    """Extract tags from prediction results.

    Args:
        probs: Prediction probability tensor
        labels: Label information
        gen_threshold: Threshold for general tags
        char_threshold: Threshold for character tags

    Returns:
        Tuple of (caption, taglist, rating_labels, char_labels, gen_labels)
    """
    # Convert indices and probabilities to labels
    label_probs: list[tuple[str, float]] = [
        (label, float(score)) for label, score in zip(labels.names, probs.numpy(), strict=False)
    ]

    # Rating labels
    rating_labels = {
        label_probs[int(idx)][0]: label_probs[int(idx)][1] for idx in labels.rating_indices
    }

    # General tags (those exceeding threshold)
    general_candidates = [
        label_probs[int(idx)]
        for idx in labels.general_indices
        if label_probs[int(idx)][1] > gen_threshold
    ]
    general_candidates.sort(key=lambda item: item[1], reverse=True)
    gen_labels = dict(general_candidates)

    # Character tags (those exceeding threshold)
    character_candidates = [
        label_probs[int(idx)]
        for idx in labels.character_indices
        if label_probs[int(idx)][1] > char_threshold
    ]
    character_candidates.sort(key=lambda item: item[1], reverse=True)
    char_labels = dict(character_candidates)

    # Combine and sort general and character tags
    combined_candidates = general_candidates + character_candidates
    combined_names = [label for label, _ in combined_candidates]

    # Convert to string usable as training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", r"\(").replace(")", r"\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


@app.command()
def infer(
    image_file: Annotated[Path, typer.Argument(help="Path to the image file for inference")],
    model_id_or_path: Annotated[
        str,
        typer.Option(
            help=(
                "Hugging Face Hub model ID or local path to a PEFT model adapter or a base model."
            ),
        ),
    ] = "SmilingWolf/wd-eva02-large-tagger-v3",
    gen_threshold: Annotated[
        float,
        typer.Option(help="Threshold for general tags", min=0.0, max=1.0),
    ] = 0.35,
    char_threshold: Annotated[
        float,
        typer.Option(help="Threshold for character tags", min=0.0, max=1.0),
    ] = 0.75,
    token: Annotated[str | None, typer.Option(help="Hugging Face API token")] = None,
    discard_existing_tags: Annotated[
        bool,
        typer.Option(
            "--discard-existing-tags/--keep-existing-tags",
            help="Discard any stored tags instead of merging with current predictions",
        ),
    ] = False,
) -> None:
    """Execute tag inference on an image."""
    # Validate image path
    image_path = Path(image_file).resolve()
    if not image_path.is_file():
        typer.echo(f"Error: Image file not found: {image_path}", err=True)
        raise typer.Exit(1)

    model: TimmWrapperForImageClassification | PeftModel
    transform: TimmWrapperImageProcessor
    typer.echo(f"Loading model and processor from {model_id_or_path}...")
    # trust_remote_code=True allows loading our custom processor
    transform = AutoImageProcessor.from_pretrained(
        model_id_or_path,
        token=token,
        trust_remote_code=True,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_id_or_path,
        token=token,
        trust_remote_code=True,
    )

    typer.echo("Loading tag list from Hub...")
    labels = load_labels_from_hub(repo_id=model_id_or_path, token=token)

    typer.echo("Loading image and preprocessing...")
    img_input = Image.open(str(image_path))
    # The custom processor now handles padding, resizing, and BGR conversion
    inputs = transform(images=img_input, return_tensors="pt")["pixel_values"]

    typer.echo("Running inference...")
    with torch.inference_mode():
        if torch_device.type != "cpu":
            model = cast("Any", model).to(device=torch_device)
            inputs = inputs.to(torch_device)

        model_output = cast("ImageClassifierOutput", model(pixel_values=inputs))
        logits = model_output.logits
        if logits is None:
            msg = "Model output did not include logits."
            raise RuntimeError(msg)
        outputs = torch.sigmoid(logits)

        if torch_device.type != "cpu":
            inputs = inputs.to("cpu")
            outputs = outputs.to("cpu")
            model = cast("Any", model).to(device="cpu")

    typer.echo("Processing results...")
    if discard_existing_tags:
        typer.echo("Discarding stored tags (merge behaviour will be wired in a future update).")
    else:
        typer.echo("Keeping stored tags when merge support lands.")
    caption, taglist, ratings, character, general = get_tags(
        probs=outputs.squeeze(0),
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


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
