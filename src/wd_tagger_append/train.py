"""CLI for LoRA fine-tuning of WD tagger models using Hugging Face Trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn, cast

import torch
import typer
from huggingface_hub.errors import HfHubHTTPError
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import Tensor, nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    BitsAndBytesConfig,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
from typer import BadParameter

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from wd_tagger_append.augmentation import (
    BatchItem,
    create_eval_transform,
    create_mixup_collate_fn,
    create_train_transform,
)
from wd_tagger_append.dataset_utils import (
    create_label_mapping,
    create_transform_function,
    determine_tag_categories,
    load_allowed_tags,
    save_labels_as_csv,
)
from wd_tagger_append.infer import MODEL_REPO_MAP, load_labels_hf
from wd_tagger_append.metrics import DEFAULT_THRESHOLD, create_compute_metrics_fn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

app = typer.Typer(help="Fine-tune WD tagger backbones with LoRA adapters.")


@dataclass
class DatasetSplits:
    """Container for train and validation splits."""

    train: Dataset
    eval: Dataset | None


def _raise_bad_parameter(message: str, cause: Exception | None = None) -> NoReturn:
    """Raise BadParameter with pyright-friendly typing."""
    error = cast("Exception", BadParameter(message))
    if cause is not None:
        error.__cause__ = cause
    raise error  # pyright: ignore[reportGeneralTypeIssues]


def _validate_dataset_inputs(dataset_path: Path | None, dataset_name: str | None) -> None:
    """Ensure exactly one dataset source is provided."""
    if dataset_path is None and dataset_name is None:
        msg = "Provide either --dataset-path or --dataset-name."
        _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]
    if dataset_path is not None and dataset_name is not None:
        msg = "Use only one of --dataset-path or --dataset-name."
        _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]


def _load_dataset_from_source(
    dataset_path: Path | None,
    dataset_name: str | None,
    dataset_config: str | None,
    token: str | None,
) -> Dataset | DatasetDict:
    """Load dataset from disk or the Hugging Face Hub."""
    if dataset_path is not None:
        typer.echo(f"Loading dataset from disk: {dataset_path}")
        return cast("Dataset | DatasetDict", load_from_disk(str(dataset_path)))

    if dataset_name is None:
        msg = "Dataset name must be provided when dataset path is not used."
        _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]

    typer.echo(f"Loading dataset from Hugging Face Hub: {dataset_name}")
    try:
        dataset = load_dataset(path=dataset_name, name=dataset_config, token=token)
        return cast("Dataset | DatasetDict", dataset)
    except HfHubHTTPError as exc:
        msg = f"Failed to download dataset {dataset_name} (config={dataset_config})"
        _raise_bad_parameter(msg, cause=exc)  # pyright: ignore[reportGeneralTypeIssues]


def _select_dataset_splits(
    dataset: Dataset | DatasetDict,
    train_split: str,
    eval_split: str | None,
) -> DatasetSplits:
    """Select train and evaluation splits."""
    if isinstance(dataset, DatasetDict):
        if train_split not in dataset:
            msg = (
                f"Train split '{train_split}' not present in dataset. "
                f"Available: {list(dataset.keys())}"
            )
            _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]
        train_ds = dataset[train_split]
        eval_ds = None
        if eval_split is not None:
            if eval_split not in dataset:
                msg = (
                    f"Eval split '{eval_split}' not present in dataset. "
                    f"Available: {list(dataset.keys())}"
                )
                _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]
            eval_ds = dataset[eval_split]
        return DatasetSplits(train=train_ds, eval=eval_ds)

    typer.echo(
        "Dataset is single-split. Using it as the training set; evaluation will be skipped.",
        err=True,
    )
    return DatasetSplits(train=dataset, eval=None)


def _merge_label_lists(
    base_labels: Sequence[str],
    dataset_labels: Iterable[str],
) -> list[str]:
    """Merge base labels with dataset-specific labels preserving base order."""
    base_set = set(base_labels)
    merged = list(base_labels)
    new_labels = sorted(label for label in dataset_labels if label not in base_set)
    merged.extend(new_labels)
    return merged


def _wrap_transform(
    transform: Callable,
    label_mapping: dict[str, int],
    categories: Sequence[str],
) -> Callable:
    """Wrap dataset transform to drop unused metadata columns."""
    base_transform = create_transform_function(transform, label_mapping, categories)

    def _transform(examples: dict) -> dict:
        processed = base_transform(examples)
        for column in ["tags", "rating", "score", "source", "md5"]:
            processed.pop(column, None)
        return processed

    return _transform


def _resolve_classifier_module(model: AutoModelForImageClassification) -> tuple[nn.Linear, str]:
    """Return classifier module and its dotted path."""
    head = model.timm_model.head  # type: ignore[union-attr]
    base_path = "timm_model.head"
    if hasattr(head, "fc"):
        if isinstance(head.fc, nn.Linear):  # type: ignore[attr-defined]
            return head.fc, f"{base_path}.fc"
        msg = f"Classifier head fc is not Linear: {type(head.fc)}"  # type: ignore[attr-defined]
        raise ValueError(msg)
    if isinstance(head, nn.Linear):
        return head, base_path
    msg = f"Unsupported classifier module type: {type(head)}"
    raise ValueError(msg)


def _expand_classification_head(
    model: AutoModelForImageClassification,
    base_labels: Sequence[str],
    target_labels: Sequence[str],
    dtype: torch.dtype | None,
) -> None:
    """Expand classification head to match target labels, copying known weights."""
    classifier, classifier_path = _resolve_classifier_module(model)
    in_features = classifier.in_features  # type: ignore[assignment]
    old_out_features = classifier.out_features  # type: ignore[assignment]
    new_out_features = len(target_labels)
    if old_out_features == new_out_features:
        typer.echo("Classification head already matches target label count.")
        return

    typer.echo(
        f"Expanding classification head ({classifier_path}) from "
        f"{old_out_features} -> {new_out_features} outputs.",
    )

    device = classifier.weight.device  # type: ignore[assignment]
    use_bias = classifier.bias is not None  # type: ignore[assignment]
    new_head = nn.Linear(in_features, new_out_features, bias=use_bias)

    if dtype is not None:
        new_head = new_head.to(dtype=dtype)
    new_head = new_head.to(device)

    base_index = {label: idx for idx, label in enumerate(base_labels)}
    with torch.no_grad():
        for target_idx, label in enumerate(target_labels):
            if label in base_index and base_index[label] < old_out_features:
                source_idx = base_index[label]
                new_head.weight[target_idx].copy_(classifier.weight[source_idx])  # type: ignore[index]
                if use_bias:
                    new_head.bias[target_idx].copy_(classifier.bias[source_idx])  # type: ignore[index]
            else:
                nn.init.normal_(new_head.weight[target_idx], mean=0.0, std=0.02)
                if use_bias:
                    nn.init.zeros_(new_head.bias[target_idx])

    head = model.timm_model.head  # type: ignore[union-attr]
    if hasattr(head, "fc"):
        head.fc = new_head
    else:
        model.timm_model.head = new_head  # type: ignore[assignment]


def _create_quantization_config(
    load_in_4bit: bool,
    load_in_8bit: bool,
    compute_dtype: torch.dtype | None,
) -> BitsAndBytesConfig | None:
    """Create BitsAndBytes quantization config when requested."""
    if not load_in_4bit and not load_in_8bit:
        return None
    if load_in_4bit and load_in_8bit:
        msg = "Choose only one of --load-in-4bit or --load-in-8bit."
        _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]
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


def _parse_precision(precision: str | None) -> tuple[bool, bool, torch.dtype | None]:
    """Parse precision flag into trainer arguments and dtype."""
    if precision is None or precision.lower() == "fp32":
        return False, False, None
    if precision.lower() == "bf16":
        return True, False, torch.bfloat16
    if precision.lower() == "fp16":
        return False, True, torch.float16
    msg = "Precision must be one of: fp32, bf16, fp16."
    _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]


def _create_data_collator(num_labels: int, mixup_alpha: float) -> Callable:
    """Create data collator for Trainer."""
    if mixup_alpha > 0:
        mixup_fn = create_mixup_collate_fn(num_classes=num_labels, alpha=mixup_alpha)

        def collator(batch: list[BatchItem]) -> dict[str, Tensor]:
            pixel_values, labels = mixup_fn(batch)
            return {"pixel_values": pixel_values, "labels": labels}

        return collator

    return DefaultDataCollator(return_tensors="pt")


@app.command()
def main(
    dataset_path: Annotated[
        Path | None,
        typer.Option(
            "--dataset-path",
            help="Path to a dataset prepared with wd-tagger-prepare.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    dataset_name: Annotated[
        str | None,
        typer.Option(
            "--dataset-name",
            help="Hugging Face dataset repository to load.",
        ),
    ] = None,
    dataset_config: Annotated[
        str | None,
        typer.Option("--dataset-config", help="Optional dataset config name."),
    ] = None,
    min_tag_count: Annotated[
        int,
        typer.Option(
            "--min-tag-count",
            min=1,
            help="Minimum occurrence count required for dataset tags.",
        ),
    ] = 1,
    rating: Annotated[
        bool,
        typer.Option(
            "--rating/--no-rating",
            help="Include rating tags when building training labels.",
        ),
    ] = True,
    general: Annotated[
        bool,
        typer.Option(
            "--general/--no-general",
            help="Include general tags when building training labels.",
        ),
    ] = True,
    character: Annotated[
        bool,
        typer.Option(
            "--character/--no-character",
            help="Include character tags when building training labels.",
        ),
    ] = True,
    allowed_tags_file: Annotated[
        Path | None,
        typer.Option(
            "--allowed-tags-file",
            help="Optional newline-delimited file restricting tags used for training.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    train_split: Annotated[
        str,
        typer.Option("--train-split", help="Split name used for training."),
    ] = "train",
    eval_split: Annotated[
        str | None,
        typer.Option(
            "--eval-split",
            help="Split name used for evaluation. Set to '' to disable.",
        ),
    ] = "validation",
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory to store checkpoints and artifacts.",
            resolve_path=True,
        ),
    ] = Path("output/lora"),
    model_key: Annotated[
        str,
        typer.Option(
            "--model",
            case_sensitive=False,
            help=f"Model key from MODEL_REPO_MAP: {', '.join(MODEL_REPO_MAP.keys())}",
        ),
    ] = "eva02-large",
    base_revision: Annotated[
        str | None,
        typer.Option("--base-revision", help="Model revision to load from the Hub."),
    ] = None,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", min=0.0),
    ] = 5e-5,
    num_train_epochs: Annotated[
        float,
        typer.Option("--num-epochs", min=0.0),
    ] = 3.0,
    train_batch_size: Annotated[
        int,
        typer.Option("--train-batch-size", min=1),
    ] = 8,
    eval_batch_size: Annotated[
        int,
        typer.Option("--eval-batch-size", min=1),
    ] = 8,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option("--gradient-accumulation-steps", min=1),
    ] = 1,
    weight_decay: Annotated[
        float,
        typer.Option("--weight-decay", min=0.0),
    ] = 0.01,
    warmup_ratio: Annotated[
        float,
        typer.Option("--warmup-ratio", min=0.0, max=1.0),
    ] = 0.05,
    lora_rank: Annotated[
        int,
        typer.Option("--lora-rank", min=1),
    ] = 16,
    lora_alpha: Annotated[
        int,
        typer.Option("--lora-alpha", min=1),
    ] = 32,
    lora_dropout: Annotated[
        float,
        typer.Option("--lora-dropout", min=0.0, max=1.0),
    ] = 0.05,
    mixup_alpha: Annotated[
        float,
        typer.Option("--mixup-alpha", min=0.0),
    ] = 0.0,
    load_in_4bit: Annotated[
        bool,
        typer.Option("--load-in-4bit", help="Enable 4-bit quantized loading."),
    ] = False,
    load_in_8bit: Annotated[
        bool,
        typer.Option("--load-in-8bit", help="Enable 8-bit quantized loading."),
    ] = False,
    precision: Annotated[
        str | None,
        typer.Option(
            "--precision",
            help="Numerical precision: fp32, bf16, or fp16.",
        ),
    ] = "bf16",
    gradient_checkpointing: Annotated[
        bool,
        typer.Option("--gradient-checkpointing/--no-gradient-checkpointing"),
    ] = True,
    metrics_threshold: Annotated[
        float,
        typer.Option("--metrics-threshold", min=0.0, max=1.0),
    ] = DEFAULT_THRESHOLD,
    logging_steps: Annotated[
        int,
        typer.Option("--logging-steps", min=1),
    ] = 50,
    save_strategy: Annotated[
        Literal["no", "steps", "epoch"],
        typer.Option(
            "--save-strategy",
            help="Checkpoint save strategy.",
        ),
    ] = "epoch",
    save_total_limit: Annotated[
        int,
        typer.Option("--save-total-limit", min=1),
    ] = 2,
    eval_strategy: Annotated[
        Literal["no", "steps", "epoch"],
        typer.Option(
            "--eval-strategy",
            help="Evaluation strategy for Trainer.",
        ),
    ] = "epoch",
    seed: Annotated[
        int,
        typer.Option("--seed"),
    ] = 42,
    resume_from_checkpoint: Annotated[
        str | None,
        typer.Option("--resume-from-checkpoint", help="Path or Hub checkpoint to resume from."),
    ] = None,
    report_to: Annotated[
        list[str] | None,
        typer.Option("--report-to", help="Reporting integrations (e.g. wandb)."),
    ] = None,
    push_to_hub: Annotated[
        bool,
        typer.Option("--push-to-hub", help="Push trained adapters to the Hugging Face Hub."),
    ] = False,
    hub_model_id: Annotated[
        str | None,
        typer.Option("--hub-model-id", help="Hub repo id to push to."),
    ] = None,
    hub_token: Annotated[
        str | None,
        typer.Option("--hub-token", help="Token for private Hub repos or datasets."),
    ] = None,
    private: Annotated[
        bool,
        typer.Option("--private", help="Make the Hub repository private when pushing."),
    ] = True,
) -> None:
    """Run LoRA fine-tuning."""
    _validate_dataset_inputs(dataset_path, dataset_name)

    repo_id = MODEL_REPO_MAP.get(model_key.lower())
    if repo_id is None:
        typer.echo(
            f"Unknown model key '{model_key}'. Options: {', '.join(MODEL_REPO_MAP.keys())}",
            err=True,
        )
        raise typer.Exit(code=1)

    raw_dataset = _load_dataset_from_source(dataset_path, dataset_name, dataset_config, hub_token)
    eval_key = eval_split or None
    splits = _select_dataset_splits(raw_dataset, train_split=train_split, eval_split=eval_key)

    typer.echo(f"Training examples: {len(splits.train)}")
    if splits.eval is not None:
        typer.echo(f"Evaluation examples: {len(splits.eval)}")

    datasets_for_labels: list[Dataset] = [splits.train]
    if splits.eval is not None:
        datasets_for_labels.append(splits.eval)
        combined = concatenate_datasets(datasets_for_labels)
    else:
        combined = splits.train

    category_flags = {
        "rating": rating,
        "general": general,
        "character": character,
    }
    category_order = ("rating", "general", "character")
    selected_categories = [name for name in category_order if category_flags[name]]
    if not selected_categories:
        msg = "At least one tag category must remain enabled."
        _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]

    allow_list: set[str] | None = None
    if allowed_tags_file is not None:
        allow_list = load_allowed_tags(allowed_tags_file)
        if not allow_list:
            msg = (
                f"No tags found in {allowed_tags_file}. Provide a file containing at least one tag."
            )
            _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]

    dataset_label_mapping, tag_frequencies = create_label_mapping(
        combined,
        categories=selected_categories,
        min_count=min_tag_count,
        allowed_tags=allow_list,
    )

    if not dataset_label_mapping:
        msg = "No dataset tags remain after applying filters."
        if allow_list is not None:
            msg += " Adjust --allowed-tags-file to match dataset tags."
        if min_tag_count > 1:
            msg += " Consider lowering --min-tag-count."
        _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]

    typer.echo("Active tag categories: " + ", ".join(selected_categories))
    typer.echo(
        "Dataset tag filter -> min-count "
        f"{min_tag_count} | kept {len(dataset_label_mapping)} of {len(tag_frequencies)} tags",
    )

    if allow_list is not None:
        retained = len(dataset_label_mapping)
        missing = len(allow_list.difference(dataset_label_mapping.keys()))
        typer.echo(
            f"Allow list matched {retained} tags; {missing} from {allowed_tags_file} "
            "were not present in the dataset.",
        )

    base_label_data = load_labels_hf(repo_id=repo_id, revision=base_revision, token=hub_token)
    base_labels = base_label_data.names
    label_list = _merge_label_lists(base_labels, dataset_label_mapping.keys())
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    typer.echo(f"Base labels: {len(base_labels)} | Dataset labels: {len(dataset_label_mapping)}")
    typer.echo(f"Final label space: {len(label_list)} tags")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save labels as CSV (WD Tagger v3 compatible format)
    csv_path = output_dir / "selected_tags.csv"

    # Build category mapping
    base_general_tags = {base_labels[idx] for idx in base_label_data.general}
    base_character_tags = {base_labels[idx] for idx in base_label_data.character}
    base_rating_tags = {base_labels[idx] for idx in base_label_data.rating}

    # Determine dataset tag categories
    dataset_tag_categories = determine_tag_categories(combined, categories=selected_categories)

    # Categorize all labels
    general_tags = []
    character_tags = []
    rating_tags = []

    for tag in label_list:
        if tag in base_rating_tags:
            rating_tags.append(tag)
        elif tag in base_character_tags:
            character_tags.append(tag)
        elif tag in base_general_tags:
            general_tags.append(tag)
        else:
            # New tags from dataset: use dataset category
            dataset_category = dataset_tag_categories.get(tag, "general")
            if dataset_category == "rating":
                rating_tags.append(tag)
            elif dataset_category == "character":
                character_tags.append(tag)
            else:
                general_tags.append(tag)

    tag_categories = {
        "rating": rating_tags,
        "general": general_tags,
        "character": character_tags,
    }

    save_labels_as_csv(label_list, tag_categories, csv_path)
    typer.echo(f"Saved labels as CSV to {csv_path}")

    train_transform = create_train_transform(
        pretrained_model_name_or_path=repo_id,
    )
    train_dataset = splits.train.with_transform(
        _wrap_transform(train_transform, label2id, selected_categories),
    )
    eval_dataset = None
    if splits.eval is not None:
        eval_transform = create_eval_transform(pretrained_model_name_or_path=repo_id)
        eval_dataset = splits.eval.with_transform(
            _wrap_transform(eval_transform, label2id, selected_categories),
        )

    bf16, fp16, precision_dtype = _parse_precision(precision)
    quant_config = _create_quantization_config(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        compute_dtype=precision_dtype,
    )

    typer.echo(f"Loading pretrained model: {repo_id}")
    model = AutoModelForImageClassification.from_pretrained(
        repo_id,
        revision=base_revision,
        device_map="auto" if quant_config is not None else None,
        quantization_config=quant_config,
        dtype=precision_dtype,
    )
    image_processor = AutoImageProcessor.from_pretrained(repo_id, revision=base_revision)

    _expand_classification_head(
        model=model,
        base_labels=base_labels,
        target_labels=label_list,
        dtype=precision_dtype,
    )
    model.config.label2id = label2id
    model.config.id2label = id2label
    model.config.num_labels = len(label_list)
    model.config.problem_type = "multi_label_classification"
    model.config.use_cache = False

    if quant_config is not None:
        typer.echo("Preparing model for k-bit training.")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()

    classifier_module_name = _resolve_classifier_module(model)[1]
    modules_to_save = {classifier_module_name}

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        modules_to_save=sorted(modules_to_save),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = _create_data_collator(num_labels=len(label_list), mixup_alpha=mixup_alpha)
    compute_metrics = create_compute_metrics_fn(
        num_labels=len(label_list),
        threshold=metrics_threshold,
    )

    trainer_eval_strategy = "no" if eval_dataset is None else eval_strategy

    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "logging_steps": logging_steps,
        "eval_strategy": trainer_eval_strategy,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "bf16": bf16,
        "fp16": fp16,
        "seed": seed,
        "remove_unused_columns": False,
        "report_to": report_to,
        "push_to_hub": push_to_hub,
        "hub_model_id": hub_model_id,
        "hub_token": hub_token,
        "hub_private_repo": private,
        "gradient_checkpointing": gradient_checkpointing,
        "load_best_model_at_end": eval_dataset is not None,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
    }
    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics if eval_dataset is not None else None,
        "tokenizer": image_processor,
    }
    trainer = Trainer(**trainer_kwargs)

    typer.echo("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    typer.echo("Training completed. Saving artifacts...")

    trainer.save_model()
    image_processor.save_pretrained(output_dir)

    if push_to_hub:
        typer.echo("Pushing adapters to the Hugging Face Hub...")
        trainer.push_to_hub()

    typer.echo("All done!")


if __name__ == "__main__":
    app()
