"""CLI for LoRA fine-tuning of WD tagger models using Hugging Face Trainer."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Annotated, Any, Literal, NoReturn, cast

import torch
import typer
from huggingface_hub.errors import HfHubHTTPError
from peft import (
    LoraConfig,
    PeftModel,
    cast_mixed_precision_params,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.utils import ModulesToSaveWrapper
from torch import Tensor, nn
from transformers import (
    BitsAndBytesConfig,
    DefaultDataCollator,
    TimmWrapperForImageClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import HubStrategy, IntervalStrategy, SaveStrategy, SchedulerType
from transformers.training_args import OptimizerNames
from typer import BadParameter

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from .augmentation import (
    BatchItem,
    WDTaggerImageProcessor,
    create_mixup_collate_fn,
)
from .dataset_utils import (
    AllowedTagSpec,
    add_replacement_categories,
    apply_replacements_to_counts,
    categorize_label_list,
    count_tag_frequencies,
    create_transform_function,
    determine_tag_categories,
    filter_tags_pandas,
    load_allowed_tags,
    save_label_mapping_as_json,
    save_labels_as_csv,
)
from .inference_utils import MODEL_REPO_MAP, LabelData, load_labels_hf
from .loss import AsymmetricLossMultiLabel
from .metrics import DEFAULT_THRESHOLD, create_compute_metrics_fn

app = typer.Typer(help="Fine-tune WD tagger backbones with LoRA adapters.")


@dataclass
class DatasetSplits:
    """Container for train and validation splits."""

    train: Dataset
    eval: Dataset | None


@dataclass(frozen=True)
class DatasetOptions:
    path: Path | None
    name: str | None
    config: str | None
    token: str | None
    train_split: str
    eval_split: str | None


@dataclass(frozen=True)
class LabelSelection:
    include_rating: bool
    include_general: bool
    include_character: bool
    min_count: int
    allowed_tags_file: Path | None

    def selected_categories(self) -> tuple[str, ...]:
        categories: list[str] = []
        if self.include_rating:
            categories.append("rating")
        if self.include_general:
            categories.append("general")
        if self.include_character:
            categories.append("character")
        return tuple(categories)


@dataclass(frozen=True)
class LabelArtifacts:
    label_list: list[str]
    label2id: dict[str, int]
    id2label: dict[int, str]
    csv_path: Path
    mapping_path: Path
    category_mapping: dict[str, list[str]]
    dataset_tag_categories: dict[str, str]
    base_labels: list[str]
    categories: tuple[str, ...]
    new_tag_count: int
    tag_frequencies: Counter[str]


def _compute_class_weights(
    frequencies: Mapping[str, int],
    labels: Sequence[str],
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Create a class weight tensor aligned with the provided label order."""
    if not labels:
        return torch.tensor([], dtype=torch.float32)

    counts = torch.tensor(
        [float(frequencies.get(label, 0)) for label in labels],
        dtype=torch.float32,
    )
    total = counts.sum()
    if total <= 0:
        return torch.ones_like(counts)

    num_classes = counts.numel()
    return (total + epsilon) / (num_classes * (counts + epsilon))


class DatasetManager:
    """Handle dataset loading and split selection."""

    def __init__(self, options: DatasetOptions) -> None:
        self._options = options

    def load(self) -> Dataset | DatasetDict:
        return _load_dataset_from_source(
            self._options.path,
            self._options.name,
            self._options.config,
            self._options.token,
        )

    def create_splits(self, dataset: Dataset | DatasetDict) -> DatasetSplits:
        eval_split = self._options.eval_split or None
        return _select_dataset_splits(dataset, self._options.train_split, eval_split)

    @staticmethod
    def combine(splits: DatasetSplits) -> Dataset:
        if splits.eval is None:
            return splits.train
        datasets = [splits.train, splits.eval]
        return concatenate_datasets(datasets)


class LabelSpaceBuilder:
    """Create merged label spaces and persist mapping artifacts."""

    def __init__(
        self,
        selection: LabelSelection,
        output_dir: Path,
        report: Callable[[str], None],
        forget_base_labels: bool,
    ) -> None:
        self._selection = selection
        self._output_dir = output_dir
        self._report = report
        self._forget_base_labels = forget_base_labels

    def build(self, base_label_data: LabelData, combined_dataset: Dataset) -> LabelArtifacts:
        all_categories = ("rating", "general", "character")

        if self._forget_base_labels:
            self._report("Base label retention disabled; using dataset tags only.")
            base_labels: list[str] = []
            base_label_set: set[str] = set()
        else:
            base_labels = list(base_label_data.names)
            base_label_set = set(base_labels)

        tag_frequencies = count_tag_frequencies(combined_dataset, categories=all_categories)
        new_tags_frequencies = {
            tag: count for tag, count in tag_frequencies.items() if tag not in base_label_set
        }

        base_count = len(tag_frequencies) - len(new_tags_frequencies)
        self._report(
            "Dataset contains "
            f"{len(tag_frequencies)} unique tags: {base_count} base model tags, "
            f"{len(new_tags_frequencies)} new tags",
        )

        dataset_tag_categories = determine_tag_categories(
            combined_dataset,
            categories=all_categories,
        )

        selected_categories = self._selection.selected_categories()
        allowed_spec: AllowedTagSpec | None = None
        allowed_lookup: set[str] | None = None
        if self._selection.allowed_tags_file is not None:
            allowed_spec = load_allowed_tags(self._selection.allowed_tags_file)
            if allowed_spec.is_empty():
                msg = (
                    f"No tags found in {self._selection.allowed_tags_file}. "
                    "Provide a file containing at least one tag."
                )
                _raise_bad_parameter(msg)
            allowed_lookup = set(allowed_spec.as_collection())

        filtered_new_tags = filter_tags_pandas(
            tag_frequencies=new_tags_frequencies,
            base_label_set=base_label_set,
            tag_categories=dataset_tag_categories,
            selected_categories=selected_categories,
            min_count=self._selection.min_count,
            allowed_tags=allowed_lookup,
        )

        self._report(
            "After filtering (category, min-count, allow-list): "
            f"{len(filtered_new_tags)} new tags remain",
        )

        replacement_map = allowed_spec.replacements if allowed_spec is not None else {}
        filtered_new_tags = apply_replacements_to_counts(filtered_new_tags, replacement_map)
        dataset_tag_categories = add_replacement_categories(
            dataset_tag_categories,
            replacement_map,
        )

        label_list = _merge_label_lists(base_labels, filtered_new_tags.keys())
        label2id = {label: idx for idx, label in enumerate(label_list)}
        id2label = {idx: label for label, idx in label2id.items()}

        self._report(
            f"Final training labels: {len(base_labels)} base tags + "
            f"{len(filtered_new_tags)} new tags = {len(label_list)} total",
        )

        if self._forget_base_labels:
            base_label_indices = {"rating": set(), "character": set(), "general": set()}
        else:
            base_label_indices = {
                "rating": {base_labels[idx] for idx in base_label_data.rating},
                "character": {base_labels[idx] for idx in base_label_data.character},
                "general": {base_labels[idx] for idx in base_label_data.general},
            }

        tag_categories = categorize_label_list(
            label_list=label_list,
            base_label_indices=base_label_indices,
            dataset_tag_categories=dataset_tag_categories,
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._output_dir / "selected_tags.csv"
        mapping_path = self._output_dir / "label_mapping.json"

        save_labels_as_csv(label_list, tag_categories, csv_path)
        self._report(f"Saved labels as CSV to {csv_path}")

        save_label_mapping_as_json(label2id, mapping_path)
        self._report(f"Saved label mapping to {mapping_path}")

        return LabelArtifacts(
            label_list=label_list,
            label2id=label2id,
            id2label=id2label,
            csv_path=csv_path,
            mapping_path=mapping_path,
            category_mapping=tag_categories,
            dataset_tag_categories=dataset_tag_categories,
            base_labels=base_labels,
            categories=all_categories,
            new_tag_count=len(filtered_new_tags),
            tag_frequencies=tag_frequencies,
        )


class TransformPlanner:
    """Create dataset transforms backed by the WD image processor."""

    def __init__(
        self,
        image_processor: WDTaggerImageProcessor,
        label2id: dict[str, int],
        categories: Sequence[str],
    ) -> None:
        self._image_processor = image_processor
        self._label2id = label2id
        self._categories = tuple(categories)

    def _process(self, image: Any, train: bool) -> torch.Tensor:
        features = self._image_processor(
            images=image,
            return_tensors="pt",
            do_train_augmentations=train,
        )
        pixel_values = cast("torch.Tensor", features["pixel_values"])
        return pixel_values.squeeze(0) if pixel_values.ndim == 4 else pixel_values

    def train_transform(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        return create_transform_function(
            lambda image: self._process(image, True),
            self._label2id,
            self._categories,
        )

    def eval_transform(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        return create_transform_function(
            lambda image: self._process(image, False),
            self._label2id,
            self._categories,
        )


@dataclass(frozen=True)
class ModelPreparationResult:
    model: PeftModel
    bf16: bool
    fp16: bool
    precision_dtype: torch.dtype | None


class ModelPreparer:
    """Load base models, expand classifier heads, and attach LoRA adapters."""

    def __init__(
        self,
        model_key: str,
        quantization: Literal["none", "8bit", "4bit"],
        gradient_checkpointing: bool,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        freeze_base_labels: bool,
        report: Callable[[str], None],
    ) -> None:
        self._model_key = model_key
        self._quantization = quantization
        self._gradient_checkpointing = gradient_checkpointing
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout
        self._freeze_base_labels = freeze_base_labels
        self._report = report

    def prepare(
        self,
        repo_id: str,
        base_revision: str | None,
        label_artifacts: LabelArtifacts,
        precision: str | None,
    ) -> ModelPreparationResult:
        bf16, fp16, precision_dtype = _parse_precision(precision)

        quantization_config = None
        if self._quantization in {"8bit", "4bit"}:
            quantization_config = _create_quantization_config(
                in_4bit=(self._quantization == "4bit"),
                compute_dtype=precision_dtype,
                skip_modules=CLASSIFIER_SKIP_MODULES,
            )
            self._report(f"{self._quantization} quantization enabled.")

        self._report(f"Loading pretrained model: {repo_id}")
        pretrained_kwargs: dict[str, Any] = {
            "dtype": precision_dtype,
            "quantization_config": quantization_config,
            "problem_type": "multi_label_classification",
        }
        if base_revision is not None:
            pretrained_kwargs["revision"] = base_revision

        model = TimmWrapperForImageClassification.from_pretrained(
            repo_id,
            **pretrained_kwargs,
        )

        _expand_classification_head(
            model=model,
            base_labels=label_artifacts.base_labels,
            target_labels=label_artifacts.label_list,
            dtype=precision_dtype,
        )
        model.config.label2id = label_artifacts.label2id
        model.config.id2label = label_artifacts.id2label
        model.config.num_labels = len(label_artifacts.label_list)

        if bf16 or fp16:
            cast_mixed_precision_params(model, dtype=precision_dtype)
        elif quantization_config is not None:
            model_key_lower = self._model_key.lower()

            def get_input_embeddings(model_self: TimmWrapperForImageClassification) -> nn.Module:
                if model_key_lower in {"eva02-large", "vit-large", "vit", "swinv2"}:
                    patch_embed = cast("Any", model_self.timm_model.patch_embed)
                    return cast("nn.Module", patch_embed.proj)
                if model_key_lower == "convnext":
                    stem = cast("nn.Sequential", model_self.timm_model.stem)
                    return cast("nn.Module", stem[0])

                msg = f"get_input_embeddings not implemented for model '{model_key_lower}'"
                raise NotImplementedError(msg)

            def set_input_embeddings(
                model_self: TimmWrapperForImageClassification,
                new_module: nn.Module,
            ) -> None:
                if model_key_lower in {"eva02-large", "vit-large", "vit", "swinv2"}:
                    patch_embed = cast("Any", model_self.timm_model.patch_embed)
                    patch_embed.proj = new_module
                elif model_key_lower == "convnext":
                    stem = cast("nn.Sequential", model_self.timm_model.stem)
                    stem[0] = new_module
                else:
                    msg = f"set_input_embeddings not implemented for model '{model_key_lower}'"
                    raise NotImplementedError(msg)

            model.get_input_embeddings = MethodType(get_input_embeddings, model)
            model.set_input_embeddings = MethodType(set_input_embeddings, model)

            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self._gradient_checkpointing,
            )

        classifier_module_name = _resolve_classifier_module(model)[1]
        modules_to_save = {classifier_module_name}
        target_modules = _resolve_lora_target_modules(self._model_key.lower())

        lora_config = LoraConfig(
            r=self._lora_rank,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            target_modules=target_modules,
            modules_to_save=sorted(modules_to_save),
            bias="none",
        )
        model = cast("PeftModel", get_peft_model(model, lora_config))
        model.print_trainable_parameters()

        if self._freeze_base_labels:
            num_base_labels = len(label_artifacts.base_labels)
            num_new_labels = len(label_artifacts.label_list) - num_base_labels
            if num_base_labels == 0:
                self._report("No base labels detected; skipping gradient masking.")
            else:
                self._report(f"Freezing {num_base_labels} base labels before LoRA fine-tuning.")
                self._report(f"Training updates target {num_new_labels} new labels only.")

                classifier, _ = _resolve_peft_module(model)

                weight_hook = _create_gradient_mask_hook(num_base_labels)
                classifier.weight.register_hook(weight_hook)

                if classifier.bias is not None:
                    bias_hook = _create_gradient_mask_hook(num_base_labels)
                    classifier.bias.register_hook(bias_hook)

                self._report("Gradient masking hooks registered for base model labels")

        return ModelPreparationResult(
            model=model,
            bf16=bf16,
            fp16=fp16,
            precision_dtype=precision_dtype,
        )


CLASSIFIER_SKIP_MODULES: tuple[str, ...] = (
    "head",
    "head.fc",
    "timm_model.head",
    "timm_model.head.fc",
)

_LORA_TARGET_MODULES: dict[str, tuple[str, ...]] = {
    "eva02-large": ("q_proj", "k_proj", "v_proj", "proj", "fc1_g", "fc1_x", "fc2"),
    "swinv2": ("qkv", "proj", "fc1", "fc2"),
    "vit": ("qkv", "proj", "fc1", "fc2"),
    "vit-large": ("qkv", "proj", "fc1", "fc2"),
    "convnext": ("fc1", "fc2"),
}


def _resolve_lora_target_modules(model_key: str) -> list[str]:
    """Return LoRA target module patterns tuned per backbone."""
    patterns = _LORA_TARGET_MODULES.get(
        model_key,
        ("q_proj", "k_proj", "v_proj", "qkv", "proj", "fc1", "fc1_g", "fc1_x", "fc2"),
    )
    # Deduplicate while preserving order
    seen: set[str] = set()
    resolved: list[str] = []
    for name in patterns:
        if name in seen:
            continue
        seen.add(name)
        resolved.append(name)
    return resolved


def _raise_bad_parameter(message: str, cause: Exception | None = None) -> NoReturn:
    """Raise BadParameter with pyright-friendly typing."""
    error = BadParameter(message)
    if cause is not None:
        error.__cause__ = cause
    raise error


def _validate_dataset_inputs(dataset_path: Path | None, dataset_name: str | None) -> None:
    """Ensure exactly one dataset source is provided."""
    if dataset_path is None and dataset_name is None:
        msg = "Provide either --dataset-path or --dataset-name."
        _raise_bad_parameter(msg)
    if dataset_path is not None and dataset_name is not None:
        msg = "Use only one of --dataset-path or --dataset-name."
        _raise_bad_parameter(msg)


def _validate_hub_inputs(
    push_to_hub: bool,
    hub_model_id: str | None,
    output_dir: Path,
) -> None:
    """Validate Hub-related options.

    When push_to_hub is True:
    - hub_model_id must be provided
    - If output_dir exists, it must be a valid git repository clone
    """
    if push_to_hub and hub_model_id is None:
        msg = "--hub-model-id is required when --push-to-hub is enabled."
        _raise_bad_parameter(msg)

    if push_to_hub and output_dir.exists():
        git_dir = output_dir / ".git"
        if not git_dir.exists():
            msg = (
                f"When --push-to-hub is enabled and --output-dir '{output_dir}' exists, "
                "it must be a valid git repository clone (containing .git directory)."
            )
            _raise_bad_parameter(msg)


def _load_dataset_from_source(
    dataset_path: Path | None,
    dataset_name: str | None,
    dataset_config: str | None,
    token: str | None,
) -> Dataset | DatasetDict:
    """Load dataset from disk or the Hugging Face Hub."""
    if dataset_path is not None:
        typer.echo(f"Loading dataset from disk: {dataset_path}")
        return load_from_disk(str(dataset_path))

    if dataset_name is None:
        msg = "Dataset name must be provided when dataset path is not used."
        _raise_bad_parameter(msg)

    typer.echo(f"Loading dataset from Hugging Face Hub: {dataset_name}")
    try:
        loaded = load_dataset(path=dataset_name, name=dataset_config, token=token)
        if isinstance(loaded, (IterableDataset, IterableDatasetDict)):
            msg = "Iterable datasets are not supported for training."
            _raise_bad_parameter(msg)
        return loaded
    except HfHubHTTPError as exc:
        msg = f"Failed to download dataset {dataset_name} (config={dataset_config})"
        _raise_bad_parameter(msg, cause=exc)


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
            _raise_bad_parameter(msg)
        train_ds = dataset[train_split]
        eval_ds = None
        if eval_split is not None:
            if eval_split not in dataset:
                msg = (
                    f"Eval split '{eval_split}' not present in dataset. "
                    f"Available: {list(dataset.keys())}"
                )
                _raise_bad_parameter(msg)
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


def _resolve_classifier_module(model: TimmWrapperForImageClassification) -> tuple[nn.Linear, str]:
    """Return classifier module and its dotted path."""
    head = cast("Any", model.timm_model.head)
    base_path = "timm_model.head"
    if hasattr(head, "fc"):
        fc_module = head.fc
        if isinstance(fc_module, nn.Linear):
            return fc_module, f"{base_path}.fc"
        msg = f"Classifier head fc is not Linear: {type(fc_module)}"
        raise ValueError(msg)
    if isinstance(head, nn.Linear):
        return head, base_path
    msg = f"Unsupported classifier module type: {type(head)}"
    raise ValueError(msg)


def _resolve_peft_module(model: PeftModel) -> tuple[nn.Linear, str]:
    """Return classifier module and its dotted path from a PEFT model."""
    head = model.base_model.model.timm_model.head
    if isinstance(head, ModulesToSaveWrapper):
        if isinstance(head.modules_to_save, nn.ModuleDict):
            default_module = head.modules_to_save["default"]
            if isinstance(default_module, nn.Linear):
                base_path = "base_model.model.timm_model.head"

                return default_module, f"{base_path}.modules_to_save.default"
            msg = (
                f"PEFT classifier modules_to_save['default'] is not Linear: {type(default_module)}"
            )
            raise ValueError(msg)
        msg = f"PEFT classifier modules_to_save is not ModuleDict: {type(head.modules_to_save)}"
        raise ValueError(msg)
    msg = f"PEFT classifier head is not ModulesToSaveWrapper: {type(head)}"
    raise ValueError(msg)


def _expand_classification_head(
    model: TimmWrapperForImageClassification,
    base_labels: Sequence[str],
    target_labels: Sequence[str],
    dtype: torch.dtype | None,
) -> None:
    """Expand classification head to match target labels, copying known weights."""
    classifier, classifier_path = _resolve_classifier_module(model)
    in_features = classifier.in_features
    old_out_features = classifier.out_features
    new_out_features = len(target_labels)
    if old_out_features == new_out_features:
        typer.echo("Classification head already matches target label count.")
        return

    typer.echo(
        f"Expanding classification head ({classifier_path}) from "
        f"{old_out_features} -> {new_out_features} outputs.",
    )

    device = classifier.weight.device
    use_bias = classifier.bias is not None
    new_head = nn.Linear(in_features, new_out_features, bias=use_bias)

    if dtype is not None:
        new_head = new_head.to(dtype=dtype)
    new_head = new_head.to(device)

    base_index = {label: idx for idx, label in enumerate(base_labels)}
    with torch.no_grad():
        for target_idx, label in enumerate(target_labels):
            if label in base_index and base_index[label] < old_out_features:
                source_idx = base_index[label]
                new_head.weight[target_idx].copy_(classifier.weight[source_idx])
                if use_bias:
                    new_head.bias[target_idx].copy_(classifier.bias[source_idx])
            else:
                nn.init.normal_(new_head.weight[target_idx], mean=0.0, std=0.02)
                if use_bias:
                    nn.init.zeros_(new_head.bias[target_idx])

    head = cast("Any", model.timm_model.head)
    if hasattr(head, "fc"):
        head.fc = new_head
    else:
        model.timm_model.head = new_head


def _parse_precision(precision: str | None) -> tuple[bool, bool, torch.dtype | None]:
    """Parse precision flag into trainer arguments and dtype."""
    bf16_supported = torch.cuda.is_bf16_supported()
    if precision is None or precision.lower() == "fp32":
        return False, False, None
    if precision.lower() == "bf16":
        if not bf16_supported:
            msg = "bfloat16 is not supported on this device."
            raise typer.BadParameter(msg)
        return True, False, torch.bfloat16
    if precision.lower() == "fp16":
        return False, True, torch.float16
    msg = "Precision must be one of: fp32, bf16, fp16."
    _raise_bad_parameter(msg)


def _create_quantization_config(
    in_4bit: bool,
    compute_dtype: torch.dtype | None,
    skip_modules: Sequence[str],
) -> BitsAndBytesConfig:
    """Construct a BitsAndBytes quantization configuration if requested."""
    if in_4bit:
        # Four-bit quantization reduces numeric precision. This config loads parameters in 4-bit
        # using nf4 with double quantization. Evaluate task-specific impact separately
        # (e.g., scripts/quantization_benchmark.py) and choose precision accordingly.
        typer.echo(
            "Caution: 4-bit quantization reduces numeric precision. "
            "Use primarily for testing. "
            "Prefer 8-bit or fp16/bf16 when accuracy sensitivity matters.",
            err=True,
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=list(skip_modules),
        llm_int8_has_fp16_weight=False,
    )


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
                "Optional file restricting NEW tags used for training. "
                "Supports newline-delimited entries or CSV with an optional replacement column. "
                "Base model tags are always retained."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    forget_base_labels: Annotated[
        bool,
        typer.Option(
            "--forget-base-labels",
            help="Exclude base model labels when building the training label space.",
        ),
    ] = False,
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
        typer.Option("--learning-rate", min=0.0, help="Initial learning rate for the optimizer."),
    ] = 5e-5,
    num_train_epochs: Annotated[
        float,
        typer.Option(
            "--num-epochs",
            min=0.0,
            help="Total number of training epochs (can be fractional).",
        ),
    ] = 3.0,
    train_batch_size: Annotated[
        int,
        typer.Option("--train-batch-size", min=1, help="Per-device training batch size."),
    ] = 8,
    eval_batch_size: Annotated[
        int,
        typer.Option("--eval-batch-size", min=1, help="Per-device evaluation batch size."),
    ] = 8,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(
            "--gradient-accumulation-steps",
            min=1,
            help="Number of steps to accumulate gradients before updating.",
        ),
    ] = 1,
    weight_decay: Annotated[
        float,
        typer.Option("--weight-decay", min=0.0, help="Weight decay (L2) applied by the optimizer."),
    ] = 0.01,
    warmup_ratio: Annotated[
        float,
        typer.Option(
            "--warmup-ratio",
            min=0.0,
            max=1.0,
            help="Fraction of total steps used for linear learning rate warmup.",
        ),
    ] = 0.05,
    lora_rank: Annotated[
        int,
        typer.Option("--lora-rank", min=1, help="Rank (r) for LoRA adapters."),
    ] = 16,
    lora_alpha: Annotated[
        int,
        typer.Option("--lora-alpha", min=1, help="LoRA alpha scaling factor."),
    ] = 32,
    lora_dropout: Annotated[
        float,
        typer.Option(
            "--lora-dropout",
            min=0.0,
            max=1.0,
            help="Dropout probability inside LoRA adapters.",
        ),
    ] = 0.05,
    mixup_alpha: Annotated[
        float,
        typer.Option(
            "--mixup-alpha",
            min=0.0,
            max=1.0,
            help="Alpha parameter for MixUp augmentation (0 to disable).",
        ),
    ] = 0.0,
    precision: Annotated[
        Literal["fp32", "bf16", "fp16"] | None,
        typer.Option(
            "--precision",
            help="Numerical precision: fp32, bf16, or fp16.",
        ),
    ] = "fp32",
    quantization: Annotated[
        Literal["none", "8bit", "4bit"],
        typer.Option(
            "--quantization",
            help="Quantization method to use.",
        ),
    ] = "none",
    gradient_checkpointing: Annotated[
        bool,
        typer.Option(
            "--gradient-checkpointing",
            help="Enable gradient checkpointing to reduce memory usage at the cost of compute.",
        ),
    ] = False,
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
        typer.Option(
            "--metrics-threshold",
            min=0.0,
            max=1.0,
            help="Threshold used when computing binary predictions for metrics.",
        ),
    ] = DEFAULT_THRESHOLD,
    logging_steps: Annotated[
        int,
        typer.Option(
            "--logging-steps",
            min=1,
            help="Number of update steps between two logs when logging_strategy='steps'.",
        ),
    ] = 50,
    logging_first_step: Annotated[
        bool,
        typer.Option("--logging-first-step"),
    ] = False,
    logging_nan_inf_filter: Annotated[
        bool,
        typer.Option("--logging-nan-inf-filter/--no-logging-nan-inf-filter"),
    ] = True,
    logging_strategy: Annotated[
        IntervalStrategy,
        typer.Option("--logging-strategy", help="Logging strategy: 'no', 'steps', or 'epoch'."),
    ] = IntervalStrategy.STEPS,
    save_strategy: Annotated[
        SaveStrategy,
        typer.Option(
            "--save-strategy",
            help="Checkpoint save strategy.",
        ),
    ] = SaveStrategy.EPOCH,
    save_total_limit: Annotated[
        int,
        typer.Option(
            "--save-total-limit",
            min=1,
            help="Maximum number of checkpoints to keep (older ones deleted).",
        ),
    ] = 2,
    eval_strategy: Annotated[
        IntervalStrategy,
        typer.Option(
            "--eval-strategy",
            help="Evaluation strategy for Trainer.",
        ),
    ] = IntervalStrategy.EPOCH,
    eval_accumulation_steps: Annotated[
        int | None,
        typer.Option(
            "--eval-accumulation-steps",
            min=1,
            help="Number of prediction steps to accumulate before moving results to CPU.",
        ),
    ] = None,
    torch_empty_cache_steps: Annotated[
        int | None,
        typer.Option(
            "--torch-empty-cache-steps",
            min=1,
            help="Call torch.empty_cache() every N steps to reduce peak GPU memory.",
        ),
    ] = None,
    adam_beta1: Annotated[
        float,
        typer.Option("--adam-beta1", min=0.0, max=1.0, help="Beta1 parameter for Adam optimizer."),
    ] = 0.9,
    adam_beta2: Annotated[
        float,
        typer.Option("--adam-beta2", min=0.0, max=1.0, help="Beta2 parameter for Adam optimizer."),
    ] = 0.999,
    adam_epsilon: Annotated[
        float,
        typer.Option(
            "--adam-epsilon",
            min=0.0,
            help="Epsilon parameter for Adam optimizer numerical stability.",
        ),
    ] = 1e-8,
    max_grad_norm: Annotated[
        float,
        typer.Option("--max-grad-norm", min=0.0, help="Max norm for gradient clipping."),
    ] = 1.0,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            min=-1,
            help="If >0, overrides num_epochs and stops after this many optimizer steps.",
        ),
    ] = -1,
    warmup_steps: Annotated[
        int,
        typer.Option(
            "--warmup-steps",
            min=0,
            help="Number of warmup steps for the LR scheduler (overrides warmup-ratio if >0).",
        ),
    ] = 0,
    lr_scheduler_type: Annotated[
        SchedulerType,
        typer.Option(
            "--lr-scheduler-type",
            help="LR scheduler type (e.g. 'linear', 'cosine', 'constant').",
        ),
    ] = SchedulerType.LINEAR,
    dataloader_drop_last: Annotated[
        bool,
        typer.Option(
            "--dataloader-drop-last",
            help="Drop last incomplete batch each epoch when True.",
        ),
    ] = False,
    dataloader_num_workers: Annotated[
        int,
        typer.Option(
            "--dataloader-num-workers",
            min=0,
            help="Number of subprocesses for data loading.",
        ),
    ] = 0,
    dataloader_pin_memory: Annotated[
        bool,
        typer.Option(
            "--dataloader-pin-memory",
            help="Pin memory in data loaders for faster host->device transfer.",
        ),
    ] = False,
    dataloader_persistent_workers: Annotated[
        bool,
        typer.Option(
            "--dataloader-persistent-workers",
            help="Keep data loader workers alive between epochs (may speed up training).",
        ),
    ] = False,
    label_smoothing_factor: Annotated[
        float,
        typer.Option(
            "--label-smoothing-factor",
            min=0.0,
            max=1.0,
            help="Label smoothing factor applied to targets during loss computation.",
        ),
    ] = 0.0,
    optim: Annotated[
        OptimizerNames,
        typer.Option("--optim", help="Optimizer to use (see transformers.optimizers)."),
    ] = OptimizerNames.ADAMW_TORCH,
    noise_level: Annotated[
        int,
        typer.Option(
            "--noise-level",
            min=0,
            max=2,
            help="Noise level for data augmentation.",
        ),
    ] = 0,
    torch_compile: Annotated[
        bool,
        typer.Option(
            "--torch-compile",
            help="Compile model with torch.compile() when available.",
        ),
    ] = False,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for reproducibility."),
    ] = 42,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-from-checkpoint",
            help="Path or Hub checkpoint to resume from.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
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
    hub_strategy: Annotated[
        HubStrategy,
        typer.Option(
            "--hub-strategy",
            help="When to push to the Hub: 'end', 'every_save', or 'checkpoint'.",
        ),
    ] = HubStrategy.EVERY_SAVE,
    hub_token: Annotated[
        str | None,
        typer.Option("--hub-token", help="Token for private Hub repos or datasets."),
    ] = None,
    private: Annotated[
        bool,
        typer.Option("--private/--public", help="Make the Hub repository private when pushing."),
    ] = True,
) -> None:
    """Run LoRA fine-tuning."""
    _validate_dataset_inputs(dataset_path, dataset_name)
    _validate_hub_inputs(push_to_hub, hub_model_id, output_dir)

    repo_id = MODEL_REPO_MAP.get(model_key.lower())
    if repo_id is None:
        typer.echo(
            f"Unknown model key '{model_key}'. Options: {', '.join(MODEL_REPO_MAP.keys())}",
            err=True,
        )
        raise typer.Exit(code=1)

    dataset_options = DatasetOptions(
        path=dataset_path,
        name=dataset_name,
        config=dataset_config,
        token=hub_token,
        train_split=train_split,
        eval_split=eval_split or None,
    )
    dataset_manager = DatasetManager(dataset_options)
    raw_dataset = dataset_manager.load()
    splits = dataset_manager.create_splits(raw_dataset)

    typer.echo(f"Training examples: {len(splits.train)}")
    if splits.eval is not None:
        typer.echo(f"Evaluation examples: {len(splits.eval)}")

    combined = DatasetManager.combine(splits)

    if forget_base_labels:
        typer.echo("Forgetting base labels; classifier head will use dataset tags only.")
        base_label_data = LabelData(names=[], rating=[], general=[], character=[])
    else:
        base_label_data = load_labels_hf(
            repo_id=repo_id,
            revision=base_revision,
            token=hub_token,
        )

    label_selection = LabelSelection(
        include_rating=rating,
        include_general=general,
        include_character=character,
        min_count=min_tag_count,
        allowed_tags_file=allowed_tags_file,
    )
    label_builder = LabelSpaceBuilder(
        label_selection,
        output_dir,
        typer.echo,
        forget_base_labels=forget_base_labels,
    )
    label_artifacts = label_builder.build(base_label_data, combined)

    typer.echo(
        f"Base labels: {len(label_artifacts.base_labels)} | "
        f"Dataset labels: {len(label_artifacts.label_list)}",
    )
    typer.echo(f"Final label space: {len(label_artifacts.label_list)} tags")

    typer.echo("Initializing image processor pipelines...")
    image_processor = WDTaggerImageProcessor(
        pretrained_model_name_or_path=repo_id,
        noise_level=noise_level,
    )

    transform_planner = TransformPlanner(
        image_processor=image_processor,
        label2id=label_artifacts.label2id,
        categories=label_artifacts.categories,
    )
    train_transform = transform_planner.train_transform()
    eval_transform = transform_planner.eval_transform()

    train_dataset = splits.train.with_transform(train_transform)
    eval_dataset = None
    if splits.eval is not None:
        eval_dataset = splits.eval.with_transform(eval_transform)

    model_preparer = ModelPreparer(
        model_key=model_key,
        quantization=quantization,
        gradient_checkpointing=gradient_checkpointing,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        freeze_base_labels=freeze_base_labels,
        report=typer.echo,
    )
    model_result = model_preparer.prepare(
        repo_id=repo_id,
        base_revision=base_revision,
        label_artifacts=label_artifacts,
        precision=precision,
    )
    model = model_result.model
    bf16 = model_result.bf16
    fp16 = model_result.fp16

    data_collator = _create_data_collator(
        num_labels=len(label_artifacts.label_list),
        mixup_alpha=mixup_alpha,
    )

    compute_metrics = create_compute_metrics_fn(
        num_labels=len(label_artifacts.label_list),
        threshold=metrics_threshold,
    )

    class_weights = _compute_class_weights(
        label_artifacts.tag_frequencies,
        label_artifacts.label_list,
    )
    if class_weights.numel() > 0:
        typer.echo("Class weights derived from dataset label frequencies.")

    loss_module = AsymmetricLossMultiLabel(class_weights=class_weights)

    trainer_eval_strategy = "no" if eval_dataset is None else eval_strategy
    load_best = eval_dataset is not None and eval_strategy != "no"

    training_args_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
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
        "hub_strategy": hub_strategy,
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
        "compute_loss_func": loss_module,
        "processing_class": image_processor,
    }
    trainer = Trainer(**trainer_kwargs)

    if class_weights.numel() > 0:
        class_weights_path = output_dir / "class_weights.pt"
        payload = {
            "labels": label_artifacts.label_list,
            "weights": class_weights.cpu(),
        }
        torch.save(payload, class_weights_path)
        typer.echo(f"Saved class weights to {class_weights_path}")

    typer.echo("Starting training...")
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    typer.echo("Training completed. Saving artifacts...")

    trainer.save_model()
    image_processor.save_pretrained(output_dir)

    typer.echo("All done!")


if __name__ == "__main__":
    app()
