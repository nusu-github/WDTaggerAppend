"""CLI for LoRA fine-tuning of WD tagger models using Hugging Face Trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn, cast

import torch
import typer
from huggingface_hub.errors import HfHubHTTPError
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import ModulesToSaveWrapper
from torch import Tensor, nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames
from typer import BadParameter

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from wd_tagger_append.augmentation import (
    BatchItem,
    create_eval_transform,
    create_mixup_collate_fn,
    create_train_transform,
)
from wd_tagger_append.dataset_utils import (
    categorize_label_list,
    create_label_mapping,
    create_transform_function,
    determine_tag_categories,
    filter_tags_pandas,
    load_allowed_tags,
    save_labels_as_csv,
)
from wd_tagger_append.infer import MODEL_REPO_MAP, load_labels_hf
from wd_tagger_append.loss import AsymmetricLossMultiLabel
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


def _resolve_peft_module(model: PeftModel) -> tuple[nn.Linear, str]:
    """Return classifier module and its dotted path from a PEFT model."""
    head = model.base_model.model.timm_model.head  # type: ignore[union-attr]
    base_path = "base_model.model.timm_model.head"

    if isinstance(head, ModulesToSaveWrapper):
        if isinstance(head.modules_to_save, nn.ModuleDict):
            default_module = head.modules_to_save["default"]
            if isinstance(default_module, nn.Linear):
                return cast("nn.Linear", default_module), f"{base_path}.modules_to_save.default"
            msg = (
                f"PEFT classifier modules_to_save['default'] is not Linear: {type(default_module)}"
            )
            raise ValueError(msg)
        msg = f"PEFT classifier modules_to_save is not ModuleDict: {type(head.modules_to_save)}"
        raise ValueError(msg)
    msg = f"PEFT classifier head is not ModulesToSaveWrapper: {type(head)}"
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


def _create_gradient_mask_hook(num_base_labels: int) -> Callable[[Tensor], Tensor]:
    """Create a gradient hook that masks gradients for base model labels.

    Prevents updates to the existing label weights during fine-tuning.

    Args:
        num_base_labels: Number of base model labels to mask

    Returns:
        Hook function that masks gradients for the first num_base_labels indices
    """

    def gradient_mask_hook(grad: Tensor) -> Tensor:
        """Zero out gradients for base model labels."""
        masked_grad = grad.clone()
        masked_grad[:num_base_labels] = 0
        return masked_grad

    return gradient_mask_hook


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
            help=(
                "Minimum occurrence count required for NEW dataset tags. "
                "Base model tags are always retained."
            ),
        ),
    ] = 1,
    rating: Annotated[
        bool,
        typer.Option(
            "--rating/--no-rating",
            help=(
                "Include NEW rating tags from dataset. Base model rating tags are always retained."
            ),
        ),
    ] = True,
    general: Annotated[
        bool,
        typer.Option(
            "--general/--no-general",
            help=(
                "Include NEW general tags from dataset. "
                "Base model general tags are always retained."
            ),
        ),
    ] = True,
    character: Annotated[
        bool,
        typer.Option(
            "--character/--no-character",
            help=(
                "Include NEW character tags from dataset. "
                "Base model character tags are always retained."
            ),
        ),
    ] = True,
    allowed_tags_file: Annotated[
        Path | None,
        typer.Option(
            "--allowed-tags-file",
            help=(
                "Optional newline-delimited file restricting NEW tags used for training. "
                "Base model tags are always retained."
            ),
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
        typer.Option("--mixup-alpha", min=0.0, max=1.0),
    ] = 0.0,
    precision: Annotated[
        str | None,
        typer.Option(
            "--precision",
            help="Numerical precision: fp32, bf16, or fp16.",
        ),
    ] = "fp32",
    gradient_checkpointing: Annotated[
        bool,
        typer.Option("--gradient-checkpointing/--no-gradient-checkpointing"),
    ] = True,
    freeze_base_labels: Annotated[
        bool,
        typer.Option(
            "--freeze-base-labels/--no-freeze-base-labels",
            help=(
                "Freeze base model labels during training "
                "(useful for domain-specific fine-tuning with few new tags)."
            ),
        ),
    ] = False,
    metrics_threshold: Annotated[
        float,
        typer.Option("--metrics-threshold", min=0.0, max=1.0),
    ] = DEFAULT_THRESHOLD,
    logging_steps: Annotated[
        int,
        typer.Option("--logging-steps", min=1),
    ] = 50,
    logging_first_step: Annotated[
        bool,
        typer.Option("--logging-first-step/--no-logging-first-step"),
    ] = False,
    logging_nan_inf_filter: Annotated[
        bool,
        typer.Option("--logging-nan-inf-filter/--no-logging-nan-inf-filter"),
    ] = True,
    logging_strategy: Annotated[
        Literal["no", "steps", "epoch"],
        typer.Option("--logging-strategy"),
    ] = "steps",
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
    eval_accumulation_steps: Annotated[
        int | None,
        typer.Option("--eval-accumulation-steps", min=1),
    ] = None,
    torch_empty_cache_steps: Annotated[
        int | None,
        typer.Option("--torch-empty-cache-steps", min=1),
    ] = None,
    adam_beta1: Annotated[
        float,
        typer.Option("--adam-beta1", min=0.0, max=1.0),
    ] = 0.9,
    adam_beta2: Annotated[
        float,
        typer.Option("--adam-beta2", min=0.0, max=1.0),
    ] = 0.999,
    adam_epsilon: Annotated[
        float,
        typer.Option("--adam-epsilon", min=0.0),
    ] = 1e-8,
    max_grad_norm: Annotated[
        float,
        typer.Option("--max-grad-norm", min=0.0),
    ] = 1.0,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=-1),
    ] = -1,
    warmup_steps: Annotated[
        int,
        typer.Option("--warmup-steps", min=0),
    ] = 0,
    lr_scheduler_type: Annotated[
        str,
        typer.Option("--lr-scheduler-type"),
    ] = "linear",
    dataloader_drop_last: Annotated[
        bool,
        typer.Option("--dataloader-drop-last/--no-dataloader-drop-last"),
    ] = False,
    dataloader_num_workers: Annotated[
        int,
        typer.Option("--dataloader-num-workers", min=0),
    ] = 0,
    dataloader_pin_memory: Annotated[
        bool,
        typer.Option("--dataloader-pin-memory/--no-dataloader-pin-memory"),
    ] = True,
    dataloader_persistent_workers: Annotated[
        bool,
        typer.Option("--dataloader-persistent-workers/--no-dataloader-persistent-workers"),
    ] = False,
    label_smoothing_factor: Annotated[
        float,
        typer.Option("--label-smoothing-factor", min=0.0, max=1.0),
    ] = 0.0,
    optim: Annotated[
        OptimizerNames,
        typer.Option("--optim"),
    ] = OptimizerNames.ADAMW_TORCH,
    torch_compile: Annotated[
        bool,
        typer.Option("--torch-compile/--no-torch-compile"),
    ] = False,
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

    # Load base model labels first
    base_label_data = load_labels_hf(repo_id=repo_id, revision=base_revision, token=hub_token)
    base_labels = base_label_data.names
    base_label_set = set(base_labels)

    # Extract ALL tags from dataset (no category filtering yet)
    all_categories = ("rating", "general", "character")
    _dataset_label_mapping_all, tag_frequencies_all = create_label_mapping(
        combined,
        categories=all_categories,
        min_count=1,  # No filtering yet
        allowed_tags=None,
    )

    # Separate base tags from new tags
    new_tags_frequencies = {
        tag: count for tag, count in tag_frequencies_all.items() if tag not in base_label_set
    }

    typer.echo(
        f"Dataset contains {len(tag_frequencies_all)} unique tags: "
        f"{len(tag_frequencies_all) - len(new_tags_frequencies)} base model tags, "
        f"{len(new_tags_frequencies)} new tags",
    )

    # Apply filters ONLY to new tags
    category_flags = {
        "rating": rating,
        "general": general,
        "character": character,
    }
    selected_categories = [name for name in all_categories if category_flags[name]]

    # Determine categories for new tags only
    dataset_tag_categories = determine_tag_categories(combined, categories=all_categories)

    # Load allow list if provided
    allow_list: set[str] | None = None
    if allowed_tags_file is not None:
        allow_list = load_allowed_tags(allowed_tags_file)
        if not allow_list:
            msg = (
                f"No tags found in {allowed_tags_file}. Provide a file containing at least one tag."
            )
            _raise_bad_parameter(msg)  # pyright: ignore[reportGeneralTypeIssues]

    # Filter new tags using Pandas-optimized function
    # This replaces multiple dict comprehensions with a single vectorized operation
    filtered_new_tags = filter_tags_pandas(
        tag_frequencies=new_tags_frequencies,  # pyright: ignore[reportArgumentType]
        base_label_set=base_label_set,
        tag_categories=dataset_tag_categories,
        selected_categories=selected_categories,
        min_count=min_tag_count,
        allowed_tags=allow_list,
    )

    typer.echo(
        f"After filtering (category, min-count, allow-list): "
        f"{len(filtered_new_tags)} new tags remain",
    )

    # Combine base labels with filtered new tags
    dataset_label_mapping = {**{tag: idx for idx, tag in enumerate(base_labels)}}
    dataset_label_mapping.update(dict.fromkeys(filtered_new_tags.keys(), 0))

    typer.echo(
        f"Final training labels: {len(base_labels)} base tags + "
        f"{len(filtered_new_tags)} new tags = {len(dataset_label_mapping)} total",
    )

    label_list = _merge_label_lists(base_labels, filtered_new_tags.keys())
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    typer.echo(f"Base labels: {len(base_labels)} | Dataset labels: {len(dataset_label_mapping)}")
    typer.echo(f"Final label space: {len(label_list)} tags")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save labels as CSV (WD Tagger v3 compatible format)
    csv_path = output_dir / "selected_tags.csv"

    # Build category mapping for base labels
    base_label_indices = {
        "rating": {base_labels[idx] for idx in base_label_data.rating},
        "character": {base_labels[idx] for idx in base_label_data.character},
        "general": {base_labels[idx] for idx in base_label_data.general},
    }

    # Categorize all labels using vectorized Pandas operation
    tag_categories = categorize_label_list(
        label_list=label_list,
        base_label_indices=base_label_indices,
        dataset_tag_categories=dataset_tag_categories,
    )

    save_labels_as_csv(label_list, tag_categories, csv_path)
    typer.echo(f"Saved labels as CSV to {csv_path}")

    train_transform = create_train_transform(
        pretrained_model_name_or_path=repo_id,
    )
    train_dataset = splits.train.with_transform(
        create_transform_function(train_transform, label2id, all_categories),
    )
    eval_dataset = None
    if splits.eval is not None:
        eval_transform = create_eval_transform(pretrained_model_name_or_path=repo_id)
        eval_dataset = splits.eval.with_transform(
            create_transform_function(eval_transform, label2id, all_categories),
        )

    bf16, fp16, precision_dtype = _parse_precision(precision)

    typer.echo(f"Loading pretrained model: {repo_id}")
    model = AutoModelForImageClassification.from_pretrained(
        repo_id,
        revision=base_revision,
        device_map="auto",
        dtype=precision_dtype,
        problem_type="multi_label_classification",
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

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    classifier_module_name = _resolve_classifier_module(model)[1]
    modules_to_save = {classifier_module_name}

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "qkv", "fc1", "fc2"],
        modules_to_save=sorted(modules_to_save),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Register gradient masking hooks if base labels should be frozen
    if freeze_base_labels:
        num_base_labels = len(base_labels)
        num_new_labels = len(label_list) - num_base_labels
        typer.echo(
            f"Freezing {num_base_labels} base labels; training only {num_new_labels} new labels",
        )

        # Get the classifier head (works with both timm_model.head and timm_model.head.fc)
        classifier, _ = _resolve_peft_module(model)  # pyright: ignore[reportArgumentType]

        # Create and register hooks for weight and bias
        weight_hook = _create_gradient_mask_hook(num_base_labels)
        classifier.weight.register_hook(weight_hook)

        if classifier.bias is not None:
            bias_hook = _create_gradient_mask_hook(num_base_labels)
            classifier.bias.register_hook(bias_hook)

        typer.echo("Gradient masking hooks registered for base model labels")

    data_collator = _create_data_collator(num_labels=len(label_list), mixup_alpha=mixup_alpha)

    compute_metrics = create_compute_metrics_fn(
        num_labels=len(label_list),
        threshold=metrics_threshold,
    )

    trainer_eval_strategy = "no" if eval_dataset is None else eval_strategy
    load_best = eval_dataset is not None and eval_strategy != "no"

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
        "logging_first_step": logging_first_step,
        "logging_nan_inf_filter": logging_nan_inf_filter,
        "logging_strategy": logging_strategy,
        "eval_strategy": trainer_eval_strategy,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "eval_accumulation_steps": eval_accumulation_steps,
        "torch_empty_cache_steps": torch_empty_cache_steps,
        "bf16": bf16,
        "fp16": fp16,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "adam_epsilon": adam_epsilon,
        "max_grad_norm": max_grad_norm,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": lr_scheduler_type,
        "dataloader_drop_last": dataloader_drop_last,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": dataloader_pin_memory,
        "dataloader_persistent_workers": dataloader_persistent_workers,
        "label_smoothing_factor": label_smoothing_factor,
        "optim": optim,
        "torch_compile": torch_compile,
        "seed": seed,
        "remove_unused_columns": False,
        "report_to": report_to,
        "push_to_hub": push_to_hub,
        "hub_model_id": hub_model_id,
        "hub_token": hub_token,
        "hub_private_repo": private,
        "gradient_checkpointing": gradient_checkpointing,
        "load_best_model_at_end": load_best,
        "metric_for_best_model": "auroc",
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
        "compute_loss_func": AsymmetricLossMultiLabel(),
        "processing_class": image_processor,
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
