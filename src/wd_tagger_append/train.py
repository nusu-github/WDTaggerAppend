"""CLI tool for model training."""

from pathlib import Path
from typing import Annotated

import torch
import typer
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TimmWrapperImageProcessor,
    Trainer,
    TrainingArguments,
)

from datasets import Sequence as DatasetSequence, Value
from wd_tagger_append.augmentations import AugmentationConfig
from wd_tagger_append.dataset_utils import (
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
    ModelName,
    get_model_repo_id,
    labels_to_dataframe,
    load_labels_from_hub,
)
from wd_tagger_append.model_export import (
    configure_model_for_remote,
    copy_custom_processor_code,
)

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
) -> None:
    """Execute model training."""
    # Get model repository ID
    model_repo_id = get_model_repo_id(model_name)
    typer.echo(f"Using model: {model_repo_id}")

    # Determine if dataset_source is a local folder or Hub repo
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

    # Load pretrained labels
    typer.echo("Loading pretrained labels...")
    pretrained_labels = load_labels_from_hub(repo_id=model_repo_id)
    typer.echo(f"Pretrained labels: {pretrained_labels.num_labels}")

    # Analyze new tags
    typer.echo("Analyzing new tags in dataset...")
    new_general_tags, new_character_tags = analyze_new_tags(dataset, pretrained_labels)
    num_new_classes = len(new_general_tags) + len(new_character_tags)

    typer.echo(f"Found {len(new_general_tags)} new general tags")
    typer.echo(f"Found {len(new_character_tags)} new character tags")
    typer.echo(f"Total new classes: {num_new_classes}")

    # Create extended labels
    extended_labels = pretrained_labels.extend_with_new_tags(
        new_general_tags,
        new_character_tags,
    )
    typer.echo(f"Total labels after extension: {extended_labels.num_labels}")

    typer.echo("Vectorizing labels once via dataset caching...")
    label_encoding_fn = create_label_encoding_function(extended_labels)
    updated_features = dataset.features.copy()
    updated_features["labels"] = DatasetSequence(Value("float32"))
    dataset = dataset.map(
        label_encoding_fn,
        batched=True,
        features=updated_features,
        desc="Encoding label vectors",
        load_from_cache_file=True,
    )
    removable_columns = [
        column
        for column in ("tags_general", "tags_character", "rating")
        if column in dataset.column_names
    ]
    if removable_columns:
        dataset = dataset.remove_columns(removable_columns)
    if "labels" not in dataset.column_names:
        msg = "Label vectorization failed to produce a 'labels' column."
        raise RuntimeError(msg)

    if random_crop_min_scale > random_crop_max_scale:
        msg = "random-crop-min-scale must be <= random-crop-max-scale"
        raise typer.BadParameter(msg)
    if cutout_min_ratio > cutout_max_ratio:
        msg = "cutout-min-ratio must be <= cutout-max-ratio"
        raise typer.BadParameter(msg)
    if mixup and mixup_alpha <= 0:
        msg = "mixup-alpha must be positive when MixUp is enabled"
        raise typer.BadParameter(msg)

    # Load original model
    typer.echo("Loading pretrained model...")
    original_model = AutoModelForImageClassification.from_pretrained(model_repo_id)

    # Save existing classification head weights
    num_original_labels = original_model.config.num_labels
    original_weight = original_model.timm_model.head.weight.data
    original_bias = original_model.timm_model.head.bias.data

    # Initialize new class weights to match existing distribution
    weight_std = torch.std(original_weight)
    bias_std = torch.std(original_bias)
    new_weight_rows = torch.randn(num_new_classes, original_weight.size(1)) * weight_std
    new_bias_values = torch.randn(num_new_classes) * bias_std

    # Reload model with new number of classes
    typer.echo(f"Creating model with {num_original_labels + num_new_classes} classes...")
    label2id, id2label = extended_labels.to_label_mappings()
    model = AutoModelForImageClassification.from_pretrained(
        model_repo_id,
        num_labels=num_original_labels + num_new_classes,
        ignore_mismatched_sizes=True,
    )
    configure_model_for_remote(model)

    # Manually set existing and new weights
    model.timm_model.head.weight.data = torch.cat([original_weight, new_weight_rows], dim=0)
    model.timm_model.head.bias.data = torch.cat([original_bias, new_bias_values], dim=0)

    # Configure LoRA
    typer.echo("Setting up LoRA...")
    lora_target_modules = []
    for block_index in range(lora_start_block, lora_end_block):
        lora_target_modules.extend(
            [
                f"blocks.{block_index}.attn.qkv",
                f"blocks.{block_index}.attn.proj",
                f"blocks.{block_index}.mlp.fc1",
                f"blocks.{block_index}.mlp.fc2",
            ],
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        modules_to_save=["head"],
    )

    # Wrap with PEFT model
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Set output directory
    if huggingface_account:
        final_output_dir = f"{huggingface_account}/{model_repo_id.split('/')[-1]}"
    # Auto-complete username from Hub if push_to_hub is enabled
    elif push_to_hub:
        typer.echo("Fetching username from Hugging Face API...")
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        final_output_dir = f"{username}/{model_repo_id.split('/')[-1]}"
        typer.echo(f"Auto-completed repository: {final_output_dir}")
    else:
        final_output_dir = output_dir
    typer.echo(f"Output directory: {final_output_dir}")

    # Training configuration
    training_args = TrainingArguments(
        final_output_dir,
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
        push_to_hub=False,  # We will push manually after merging
    )

    # Load image processor
    image_processor: TimmWrapperImageProcessor = AutoImageProcessor.from_pretrained(
        model_repo_id,
    )
    image_height, image_width = get_image_processor_size(image_processor)

    augmentation_config = AugmentationConfig(
        size=(image_height, image_width),
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
    eval_augmentation_config = augmentation_config.evaluation_variant()

    typer.echo("Augmentations configured:")
    typer.echo(
        f"  flip={augmentation_config.apply_flip} (p={augmentation_config.flip_prob:.2f})",
    )
    typer.echo(
        "  crop="
        f"{augmentation_config.apply_random_crop} "
        f"(scale={augmentation_config.random_crop_min_scale:.3f}"
        f"-{augmentation_config.random_crop_max_scale:.3f})",
    )
    typer.echo(
        f"  rotation={augmentation_config.apply_rotation} "
        f"(±{augmentation_config.max_rotation_degrees:.1f}°)",
    )
    typer.echo(
        "  cutout="
        f"{augmentation_config.apply_cutout} "
        f"(p={augmentation_config.cutout_prob:.2f}, "
        f"ratio={augmentation_config.cutout_min_ratio:.2f}"
        f"-{augmentation_config.cutout_max_ratio:.2f})",
    )
    mixup_alpha_value = mixup_alpha if mixup else 0.0
    typer.echo(
        f"  mixup={mixup} (alpha={mixup_alpha_value:.3f}, p={mixup_prob:.2f})",
    )

    # Split dataset
    typer.echo("Splitting dataset...")
    train_val_split = dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    # Set transformation function
    train_transform_fn = create_transform_function(
        extended_labels,
        image_processor,
        config=augmentation_config,
    )
    val_transform_fn = create_transform_function(
        extended_labels,
        image_processor,
        config=eval_augmentation_config,
    )
    train_dataset.set_transform(train_transform_fn)
    val_dataset.set_transform(val_transform_fn)

    collator = create_collate_fn(
        mixup_alpha=mixup_alpha if mixup else None,
        mixup_prob=mixup_prob,
    )

    # Create trainer and train
    typer.echo("Creating trainer...")
    trainer = Trainer(
        peft_model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        data_collator=collator,
    )

    typer.echo("Saving extended labels...")
    labels_df = labels_to_dataframe(extended_labels)
    output_csv_path = Path(training_args.output_dir) / "selected_tags.csv"
    labels_df.to_csv(output_csv_path, index=False)
    typer.echo(f"Extended labels saved to {output_csv_path}")

    typer.echo("Starting training...")
    trainer.train()

    typer.echo("Training completed!")

    if push_to_hub:
        typer.echo("Merging adapter and preparing for upload...")
        best_checkpoint_path = trainer.state.best_model_checkpoint
        if best_checkpoint_path is None:
            typer.echo("No best checkpoint found. Cannot merge and upload.", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading best adapter from {best_checkpoint_path}")

        # Reload base model for merging
        label2id, id2label = extended_labels.to_label_mappings()
        base_model = AutoModelForImageClassification.from_pretrained(
            model_repo_id,
            num_labels=extended_labels.num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )

        base_model.timm_model.head.weight.data = torch.cat(
            [original_weight, new_weight_rows],
            dim=0,
        )
        model.timm_model.head.bias.data = torch.cat([original_bias, new_bias_values], dim=0)

        # Load PEFT model and merge
        merged_model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
        merged_model = merged_model.merge_and_unload()  # pyright: ignore[reportCallIssue]

        # Update config for easy pipeline usage
        configure_model_for_remote(merged_model)

        output_dir_path = Path(training_args.output_dir)
        typer.echo(f"Saving merged model to {output_dir_path}")
        merged_model.save_pretrained(output_dir_path)

        # Save the correct processor config
        image_processor.save_pretrained(output_dir_path)

        # Copy custom processor code to output dir
        custom_processor_dest = copy_custom_processor_code(output_dir_path)
        typer.echo(f"Copied custom processor code to {custom_processor_dest}")

        # Save extended tags as well
        labels_df = labels_to_dataframe(extended_labels)
        output_csv_path = output_dir_path / "selected_tags.csv"
        labels_df.to_csv(output_csv_path, index=False)
        typer.echo(f"Extended labels saved to {output_csv_path}")

        hub_repo_id = training_args.hub_model_id or final_output_dir
        typer.echo(f"Uploading final model to {hub_repo_id}...")
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(output_dir_path),
            repo_id=hub_repo_id,
            token=token,
            repo_type="model",
        )
        typer.echo(f"Successfully uploaded merged model to https://huggingface.co/{hub_repo_id}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
