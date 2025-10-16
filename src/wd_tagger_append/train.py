"""CLI tool for model training."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import torch
import typer
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    PreTrainedModel,
    TimmWrapperImageProcessor,
    TrainingArguments,
)

from datasets import Dataset, Sequence as DatasetSequence, Value
from wd_tagger_append.augmentations import AugmentationConfig
from wd_tagger_append.constants import CUSTOM_PROCESSOR_FILENAME, DEFAULT_TAGS_FILENAME
from wd_tagger_append.dataset_utils import (
    CollatorInputItem,
    DatasetSource,
    analyze_new_tags,
    create_collate_fn,
    create_label_encoding_function,
    create_transform_function,
    detect_dataset_source,
    get_image_processor_size,
    load_dataset_with_origin,
)
from wd_tagger_append.labels import (
    ModelLabels,
    ModelName,
    get_model_repo_id,
    labels_to_dataframe,
    load_labels_from_hub,
)
from wd_tagger_append.lora_config import generate_lora_target_modules
from wd_tagger_append.model_export import (
    configure_model_for_remote,
    copy_custom_processor_code,
)
from wd_tagger_append.training import (
    ConsistencyConfig,
    ConsistencyTrainer,
    build_teacher_model,
)


@dataclass(slots=True)
class ModelInitializationArtifacts:
    """Container for model and head initialization tensors."""

    model: AutoModelForImageClassification
    original_weight: torch.Tensor
    original_bias: torch.Tensor
    new_weight_rows: torch.Tensor
    new_bias_values: torch.Tensor


def _load_and_validate_dataset(
    dataset_source: str,
    token: str | None,
) -> tuple[DatasetSource, Dataset]:
    """Load dataset and report its origin."""
    dataset_origin = detect_dataset_source(dataset_source)
    if dataset_origin is DatasetSource.LOCAL:
        typer.echo(f"Loading dataset from local folder: {dataset_source}")
    else:
        typer.echo(f"Loading dataset from Hugging Face Hub: {dataset_source}")
    _, dataset = load_dataset_with_origin(
        dataset_source,
        token=token,
        source=dataset_origin,
    )
    typer.echo(f"Loaded {len(dataset)} images")
    return dataset_origin, dataset


def _analyze_and_extend_labels(
    dataset: Dataset,
    model_repo_id: str,
    token: str | None,
) -> tuple[ModelLabels, ModelLabels, set[str], set[str]]:
    """Analyse dataset tags and extend pretrained labels."""
    typer.echo("Loading pretrained labels...")
    pretrained_labels = load_labels_from_hub(repo_id=model_repo_id, token=token)
    typer.echo(f"Pretrained labels: {pretrained_labels.num_labels}")

    typer.echo("Analyzing new tags in dataset...")
    new_general_tags, new_character_tags = analyze_new_tags(dataset, pretrained_labels)
    typer.echo(f"Found {len(new_general_tags)} new general tags")
    typer.echo(f"Found {len(new_character_tags)} new character tags")
    typer.echo(f"Total new classes: {len(new_general_tags) + len(new_character_tags)}")

    extended_labels = pretrained_labels.extend_with_new_tags(
        new_general_tags,
        new_character_tags,
    )
    typer.echo(f"Total labels after extension: {extended_labels.num_labels}")
    return pretrained_labels, extended_labels, new_general_tags, new_character_tags


def _vectorize_dataset_labels(
    dataset: Dataset,
    extended_labels: ModelLabels,
) -> Dataset:
    """Encode label vectors once and prune unused columns."""
    typer.echo("Vectorizing labels once via dataset caching...")
    label_encoding_fn = create_label_encoding_function(extended_labels)
    updated_features = dataset.features.copy()
    updated_features["labels"] = DatasetSequence(Value("bool"))
    vectorized = dataset.map(
        label_encoding_fn,
        batched=True,
        features=updated_features,
        desc="Encoding label vectors",
        load_from_cache_file=True,
    )
    removable_columns = [
        column
        for column in ("tags_general", "tags_character", "rating")
        if column in vectorized.column_names
    ]
    if removable_columns:
        vectorized = vectorized.remove_columns(removable_columns)
    if "labels" not in vectorized.column_names:
        msg = "Label vectorization failed to produce a 'labels' column."
        raise RuntimeError(msg)
    return vectorized


def _initialize_model_with_extended_head(
    model_repo_id: str,
    extended_labels: ModelLabels,
    num_new_classes: int,
) -> ModelInitializationArtifacts:
    """Load base model and resize classification head for new labels."""
    typer.echo("Loading pretrained model...")
    original_model = AutoModelForImageClassification.from_pretrained(model_repo_id)

    num_original_labels = original_model.config.num_labels
    original_weight = original_model.timm_model.head.weight.data
    original_bias = original_model.timm_model.head.bias.data

    weight_std = torch.std(original_weight)
    bias_std = torch.std(original_bias)
    new_weight_rows = torch.randn(num_new_classes, original_weight.size(1)) * weight_std
    new_bias_values = torch.randn(num_new_classes) * bias_std

    typer.echo(f"Creating model with {num_original_labels + num_new_classes} classes...")
    label2id, id2label = extended_labels.to_label_mappings()
    model = AutoModelForImageClassification.from_pretrained(
        model_repo_id,
        num_labels=num_original_labels + num_new_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    configure_model_for_remote(model)

    expanded_weight = torch.cat([original_weight, new_weight_rows], dim=0)
    expanded_bias = torch.cat([original_bias, new_bias_values], dim=0)
    model.timm_model.head.weight.data = expanded_weight.clone()
    model.timm_model.head.bias.data = expanded_bias.clone()

    return ModelInitializationArtifacts(
        model=model,
        original_weight=original_weight,
        original_bias=original_bias,
        new_weight_rows=new_weight_rows,
        new_bias_values=new_bias_values,
    )


def _create_peft_model_with_lora(
    model: AutoModelForImageClassification,
    model_name: ModelName,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_start_block: int,
    lora_end_block: int,
) -> PeftModel:
    """Create a PEFT model configured with LoRA adapters."""
    typer.echo("Setting up LoRA...")
    try:
        lora_target_modules = generate_lora_target_modules(
            model_name=model_name,
            start_block=lora_start_block,
            end_block=lora_end_block,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        modules_to_save=["head"],
    )
    base_model = cast("PreTrainedModel", model)
    peft_model = cast("PeftModel", get_peft_model(base_model, lora_config))
    peft_model.print_trainable_parameters()
    return peft_model


def _build_teacher_model_if_needed(
    *,
    consistency_weight: float,
    model_repo_id: str,
    extended_labels: ModelLabels,
    artifacts: ModelInitializationArtifacts,
) -> PreTrainedModel | None:
    """Create a frozen teacher model when consistency regularisation is enabled."""
    if consistency_weight <= 0.0:
        typer.echo("Consistency loss disabled (teacher model not created).")
        return None

    typer.echo("Building frozen teacher model for consistency regularisation...")
    return build_teacher_model(
        model_repo_id=model_repo_id,
        extended_labels=extended_labels,
        original_weight=artifacts.original_weight.detach().clone(),
        original_bias=artifacts.original_bias.detach().clone(),
        new_weight_rows=artifacts.new_weight_rows.detach().clone(),
        new_bias_values=artifacts.new_bias_values.detach().clone(),
    )


def _determine_output_directory(
    *,
    huggingface_account: str | None,
    push_to_hub: bool,
    model_repo_id: str,
    output_dir: str,
) -> tuple[str, str | None]:
    """Resolve output directory and Hub repo ID.

    Returns:
        Tuple of (local_output_dir, hub_repo_id).
        hub_repo_id is None when push_to_hub is False.
    """
    if push_to_hub:
        # For Hub upload, we use a temporary local directory
        # and determine the Hub repo ID
        local_dir = output_dir
        if huggingface_account:
            hub_repo = f"{huggingface_account}/{model_repo_id.split('/')[-1]}"
        else:
            # Use partial repo name; push_to_hub will auto-complete username
            hub_repo = model_repo_id.split("/")[-1]
            typer.echo(
                f"Hub repository will be auto-completed from partial name: {hub_repo}",
            )
        typer.echo(f"Local output directory: {local_dir}")
        typer.echo(f"Hub repository: {hub_repo}")
        return local_dir, hub_repo
    # Local-only mode
    typer.echo(f"Output directory (local only): {output_dir}")
    return output_dir, None


def _validate_mixup_params(mixup: bool, mixup_alpha: float) -> None:
    """Validate MixUp configuration parameters."""
    if mixup and mixup_alpha <= 0:
        msg = "mixup-alpha must be positive when MixUp is enabled"
        raise typer.BadParameter(msg)


def _build_augmentation_configs(
    *,
    image_size: tuple[int, int],
    random_flip: bool,
    flip_prob: float,
    random_crop: bool,
    random_crop_min_scale: float,
    random_crop_max_scale: float,
    random_rotation: bool,
    max_rotation_degrees: float,
    cutout: bool,
    cutout_prob: float,
    cutout_min_ratio: float,
    cutout_max_ratio: float,
) -> tuple[AugmentationConfig, AugmentationConfig]:
    """Create training and evaluation augmentation configurations."""
    if random_crop_min_scale > random_crop_max_scale:
        msg = "random-crop-min-scale must be <= random-crop-max-scale"
        raise typer.BadParameter(msg)
    if cutout_min_ratio > cutout_max_ratio:
        msg = "cutout-min-ratio must be <= cutout-max-ratio"
        raise typer.BadParameter(msg)

    augmentation_config = AugmentationConfig(
        size=image_size,
        apply_flip=random_flip,
        flip_prob=flip_prob,
        apply_random_crop=random_crop,
        random_crop_min_scale=random_crop_min_scale,
        random_crop_max_scale=random_crop_max_scale,
        apply_rotation=random_rotation,
        max_rotation_degrees=max_rotation_degrees,
        apply_cutout=cutout,
        cutout_prob=cutout_prob,
        cutout_min_ratio=cutout_min_ratio,
        cutout_max_ratio=cutout_max_ratio,
        random_interpolation=True,
    )
    return augmentation_config, augmentation_config.evaluation_variant()


def _log_augmentation_config(
    config: AugmentationConfig,
    *,
    mixup: bool,
    mixup_alpha: float,
    mixup_prob: float,
) -> None:
    """Log augmentation configuration details."""
    typer.echo("Augmentations configured:")
    typer.echo(
        f"  flip={config.apply_flip} (p={config.flip_prob:.2f})",
    )
    typer.echo(
        "  crop="
        f"{config.apply_random_crop} "
        f"(scale={config.random_crop_min_scale:.3f}"
        f"-{config.random_crop_max_scale:.3f})",
    )
    typer.echo(
        f"  rotation={config.apply_rotation} (±{config.max_rotation_degrees:.1f}°)",
    )
    typer.echo(
        "  cutout="
        f"{config.apply_cutout} "
        f"(p={config.cutout_prob:.2f}, "
        f"ratio={config.cutout_min_ratio:.2f}"
        f"-{config.cutout_max_ratio:.2f})",
    )

    mixup_alpha_value = mixup_alpha if mixup else 0.0
    typer.echo(
        f"  mixup={mixup} (alpha={mixup_alpha_value:.3f}, p={mixup_prob:.2f})",
    )


def _prepare_datasets_with_transforms(
    dataset: Dataset,
    *,
    seed: int,
    test_size: float,
    labels: ModelLabels,
    image_processor: TimmWrapperImageProcessor,
    train_config: AugmentationConfig,
    eval_config: AugmentationConfig,
    mixup: bool,
    mixup_alpha: float,
    mixup_prob: float,
) -> tuple[Dataset, Dataset, Callable[[list[CollatorInputItem]], dict[str, torch.Tensor]]]:
    """Split dataset, apply transforms, and create collate function."""
    typer.echo("Splitting dataset...")
    train_val_split = dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    train_transform_fn = create_transform_function(
        labels,
        image_processor,
        config=train_config,
    )
    val_transform_fn = create_transform_function(
        labels,
        image_processor,
        config=eval_config,
    )
    train_dataset.set_transform(train_transform_fn)
    val_dataset.set_transform(val_transform_fn)

    collator = create_collate_fn(
        mixup_alpha=mixup_alpha if mixup else None,
        mixup_prob=mixup_prob,
    )
    return train_dataset, val_dataset, collator


def _merge_and_push_to_hub(
    *,
    trainer: ConsistencyTrainer,
    model_repo_id: str,
    extended_labels: ModelLabels,
    artifacts: ModelInitializationArtifacts,
    image_processor: TimmWrapperImageProcessor,
    output_dir_path: Path,
    hub_repo_id: str,
    token: str | None,
    training_args: TrainingArguments,
) -> None:
    """Merge LoRA adapters and upload the resulting model to the Hub."""
    typer.echo("Merging adapter and preparing for upload...")
    best_checkpoint_path = trainer.state.best_model_checkpoint
    if best_checkpoint_path is None:
        typer.echo("No best checkpoint found. Cannot merge and upload.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading best adapter from {best_checkpoint_path}")

    label2id, id2label = extended_labels.to_label_mappings()
    base_model = AutoModelForImageClassification.from_pretrained(
        model_repo_id,
        num_labels=extended_labels.num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    base_model.timm_model.head.weight.data = torch.cat(
        [artifacts.original_weight, artifacts.new_weight_rows],
        dim=0,
    )
    base_model.timm_model.head.bias.data = torch.cat(
        [artifacts.original_bias, artifacts.new_bias_values],
        dim=0,
    )

    merged_model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
    merged_model = merged_model.merge_and_unload()  # pyright: ignore[reportCallIssue]

    configure_model_for_remote(merged_model)

    typer.echo(f"Saving merged model to {output_dir_path}")
    merged_model.save_pretrained(output_dir_path)
    image_processor.save_pretrained(output_dir_path)

    custom_processor_dest = copy_custom_processor_code(output_dir_path)
    typer.echo(f"Copied custom processor code to {custom_processor_dest}")

    labels_df = labels_to_dataframe(extended_labels)
    output_csv_path = output_dir_path / DEFAULT_TAGS_FILENAME
    labels_df.to_csv(output_csv_path, index=False)
    typer.echo(f"Extended labels saved to {output_csv_path}")

    typer.echo(f"Uploading final model to {hub_repo_id}...")

    # Use push_to_hub for the model and processor (high-level API)
    typer.echo("Pushing model to Hub...")
    merged_model.push_to_hub(
        repo_id=hub_repo_id,
        token=token,
        commit_message="Upload merged model after training",
    )

    typer.echo("Pushing processor to Hub...")
    image_processor.push_to_hub(
        repo_id=hub_repo_id,
        token=token,
        commit_message="Upload image processor",
    )
    api = HfApi(token=token)

    # Upload custom processing code and labels CSV
    custom_files_to_upload = [
        (CUSTOM_PROCESSOR_FILENAME, CUSTOM_PROCESSOR_FILENAME),
        (output_csv_path.name, DEFAULT_TAGS_FILENAME),
    ]

    for local_name, repo_path in custom_files_to_upload:
        local_file = output_dir_path / local_name
        if local_file.exists():
            typer.echo(f"Uploading {local_name} to {repo_path}...")
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=repo_path,
                repo_id=hub_repo_id,
                token=token,
                repo_type="model",
                commit_message=f"Add {local_name}",
            )

    typer.echo(f"Successfully uploaded merged model to https://huggingface.co/{hub_repo_id}")


app = typer.Typer(help="WD Tagger model training tool")


@app.command()
def train(
    dataset_source: Annotated[
        str,
        typer.Argument(
            help=(
                "Dataset source: either a local folder path or a Hugging Face "
                "dataset repo ID (e.g., username/dataset-name)"
            ),
        ),
    ],
    model_name: Annotated[
        ModelName,
        typer.Option(help="Model name to use (convnext, eva02-large, swinv2, vit-large, vit)"),
    ] = "eva02-large",
    output_dir: Annotated[
        str,
        typer.Option(help="Output directory for training results"),
    ] = "output",
    huggingface_account: Annotated[
        str | None,
        typer.Option(help="Hugging Face account name (used in output path)"),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option(help="Hugging Face API token (for private datasets)"),
    ] = None,
    push_to_hub: Annotated[
        bool,
        typer.Option(help="Push the model to the Hugging Face Hub after training"),
    ] = False,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 4,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(help="Gradient accumulation steps"),
    ] = 4,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 5e-3,
    num_epochs: Annotated[int, typer.Option(help="Number of epochs")] = 5,
    lora_r: Annotated[int, typer.Option(help="LoRA rank")] = 16,
    lora_alpha: Annotated[int, typer.Option(help="LoRA alpha value")] = 16,
    lora_dropout: Annotated[float, typer.Option(help="LoRA dropout rate")] = 0.05,
    lora_start_block: Annotated[int, typer.Option(help="Starting block for LoRA application")] = 8,
    lora_end_block: Annotated[int, typer.Option(help="Ending block for LoRA application")] = 24,
    test_size: Annotated[float, typer.Option(help="Validation data ratio", min=0.0, max=1.0)] = 0.1,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    random_flip: Annotated[
        bool,
        typer.Option(
            "--random-flip/--no-random-flip",
            help="Apply random horizontal flips during training",
        ),
    ] = True,
    flip_prob: Annotated[
        float,
        typer.Option(help="Probability of applying a horizontal flip", min=0.0, max=1.0),
    ] = 0.5,
    random_crop: Annotated[
        bool,
        typer.Option(
            "--random-crop/--no-random-crop",
            help="Apply random square crops before resizing",
        ),
    ] = True,
    random_crop_min_scale: Annotated[
        float,
        typer.Option(
            help="Minimum retained area fraction for random crop",
            min=0.0,
            max=1.0,
        ),
    ] = 0.87,
    random_crop_max_scale: Annotated[
        float,
        typer.Option(
            help="Maximum retained area fraction for random crop",
            min=0.0,
            max=1.0,
        ),
    ] = 0.998,
    random_rotation: Annotated[
        bool,
        typer.Option(
            "--random-rotation/--no-random-rotation",
            help="Apply random rotations to images",
        ),
    ] = True,
    max_rotation_degrees: Annotated[
        float,
        typer.Option(help="Maximum absolute degrees for random rotation"),
    ] = 45.0,
    cutout: Annotated[
        bool,
        typer.Option(
            "--cutout/--no-cutout",
            help="Apply random cutout masking to images",
        ),
    ] = True,
    cutout_prob: Annotated[
        float,
        typer.Option(help="Probability of applying cutout", min=0.0, max=1.0),
    ] = 0.5,
    cutout_min_ratio: Annotated[
        float,
        typer.Option(
            help="Minimum cutout size as a fraction of image size",
            min=0.0,
            max=1.0,
        ),
    ] = 0.05,
    cutout_max_ratio: Annotated[
        float,
        typer.Option(
            help="Maximum cutout size as a fraction of image size",
            min=0.0,
            max=1.0,
        ),
    ] = 0.35,
    mixup: Annotated[
        bool,
        typer.Option("--mixup/--no-mixup", help="Apply MixUp augmentation on mini-batches"),
    ] = True,
    mixup_alpha: Annotated[
        float,
        typer.Option(help="Alpha parameter for the MixUp beta distribution", min=0.0),
    ] = 0.4,
    mixup_prob: Annotated[
        float,
        typer.Option(help="Probability of applying MixUp to a batch", min=0.0, max=1.0),
    ] = 1.0,
    consistency_weight: Annotated[
        float,
        typer.Option(
            help="Weight applied to the teacher consistency KL penalty (0 disables the feature)",
            min=0.0,
        ),
    ] = 0.2,
    consistency_warmup_ratio: Annotated[
        float,
        typer.Option(
            help="Fraction of total steps reserved to warm up the consistency penalty",
            min=0.0,
            max=1.0,
        ),
    ] = 0.0,
) -> None:
    """Execute model training."""
    model_repo_id = get_model_repo_id(model_name)
    typer.echo(f"Using model: {model_repo_id}")

    _, dataset = _load_and_validate_dataset(dataset_source, token)

    (
        _,
        extended_labels,
        new_general_tags,
        new_character_tags,
    ) = _analyze_and_extend_labels(dataset, model_repo_id, token)
    dataset = _vectorize_dataset_labels(dataset, extended_labels)

    num_new_classes = len(new_general_tags) + len(new_character_tags)

    _validate_mixup_params(mixup, mixup_alpha)
    artifacts = _initialize_model_with_extended_head(
        model_repo_id,
        extended_labels,
        num_new_classes,
    )
    peft_model = _create_peft_model_with_lora(
        artifacts.model,
        model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_start_block=lora_start_block,
        lora_end_block=lora_end_block,
    )
    teacher_model = _build_teacher_model_if_needed(
        consistency_weight=consistency_weight,
        model_repo_id=model_repo_id,
        extended_labels=extended_labels,
        artifacts=artifacts,
    )

    local_output_dir, hub_repo_id = _determine_output_directory(
        huggingface_account=huggingface_account,
        push_to_hub=push_to_hub,
        model_repo_id=model_repo_id,
        output_dir=output_dir,
    )
    output_dir_path = Path(local_output_dir)
    if push_to_hub and hub_repo_id is None:
        typer.echo("Error: hub_repo_id not set despite push_to_hub=True", err=True)
        raise typer.Exit(1)

    training_args = TrainingArguments(
        str(output_dir_path),
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        num_train_epochs=num_epochs,
        logging_steps=10,
        load_best_model_at_end=True,
        label_names=["labels"],
        push_to_hub=False,
    )

    image_processor: TimmWrapperImageProcessor = AutoImageProcessor.from_pretrained(
        model_repo_id,
    )
    image_height, image_width = get_image_processor_size(image_processor)
    augmentation_config, eval_augmentation_config = _build_augmentation_configs(
        image_size=(image_height, image_width),
        random_flip=random_flip,
        flip_prob=flip_prob,
        random_crop=random_crop,
        random_crop_min_scale=random_crop_min_scale,
        random_crop_max_scale=random_crop_max_scale,
        random_rotation=random_rotation,
        max_rotation_degrees=max_rotation_degrees,
        cutout=cutout,
        cutout_prob=cutout_prob,
        cutout_min_ratio=cutout_min_ratio,
        cutout_max_ratio=cutout_max_ratio,
    )

    _log_augmentation_config(
        augmentation_config,
        mixup=mixup,
        mixup_alpha=mixup_alpha,
        mixup_prob=mixup_prob,
    )

    train_dataset, val_dataset, collator = _prepare_datasets_with_transforms(
        dataset,
        seed=seed,
        test_size=test_size,
        labels=extended_labels,
        image_processor=image_processor,
        train_config=augmentation_config,
        eval_config=eval_augmentation_config,
        mixup=mixup,
        mixup_alpha=mixup_alpha,
        mixup_prob=mixup_prob,
    )

    typer.echo("Creating trainer...")
    consistency_config = ConsistencyConfig(
        weight=consistency_weight,
        warmup_ratio=consistency_warmup_ratio,
        teacher=teacher_model,
    )
    trainer = ConsistencyTrainer(
        consistency=consistency_config,
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        data_collator=collator,
    )

    typer.echo("Saving extended labels...")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    labels_df = labels_to_dataframe(extended_labels)
    output_csv_path = output_dir_path / DEFAULT_TAGS_FILENAME
    labels_df.to_csv(output_csv_path, index=False)
    typer.echo(f"Extended labels saved to {output_csv_path}")

    typer.echo("Starting training...")
    trainer.train()
    typer.echo("Training completed!")

    if push_to_hub:
        _merge_and_push_to_hub(
            trainer=trainer,
            model_repo_id=model_repo_id,
            extended_labels=extended_labels,
            artifacts=artifacts,
            image_processor=image_processor,
            output_dir_path=output_dir_path,
            hub_repo_id=hub_repo_id,
            token=token,
            training_args=training_args,
        )


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
