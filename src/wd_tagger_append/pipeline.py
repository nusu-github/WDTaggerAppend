"""Custom Transformers pipeline for WD Tagger inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageClassification
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.image_classification import ImageClassificationPipeline

from wd_tagger_append.augmentation import WDTaggerImageProcessor
from wd_tagger_append.inference_utils import (
    MODEL_REPO_MAP,
    LabelData,
    _create_quantization_config,
    _is_local_path,
    _load_label_data,
    _parse_precision,
    _resolve_base_model_identifier,
    get_tags,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class WDTaggerPipeline(ImageClassificationPipeline):
    """Pipeline that returns caption-style outputs from WD Tagger models."""

    task = "wd-tagger"

    def __init__(
        self,
        model: torch.nn.Module,
        image_processor: WDTaggerImageProcessor,
        *,
        label_data: LabelData,
        gen_threshold: float = 0.35,
        char_threshold: float = 0.75,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            image_processor=image_processor,
            feature_extractor=image_processor,
            **kwargs,
        )
        self.model.eval()
        self._label_data = label_data
        self._default_gen_threshold = gen_threshold
        self._default_char_threshold = char_threshold

    def _sanitize_parameters(
        self,
        top_k: int | None = None,
        function_to_apply: str | None = None,
        timeout: float | None = None,
        gen_threshold: float | None = None,
        char_threshold: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        preprocess_params: dict[str, Any] = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout

        postprocess_params: dict[str, Any] = {}
        if gen_threshold is not None:
            postprocess_params["gen_threshold"] = gen_threshold
        if char_threshold is not None:
            postprocess_params["char_threshold"] = char_threshold

        # The WD tagger pipeline does not use top-k or function_to_apply,
        # but they are accepted for API compatibility.
        return preprocess_params, {}, postprocess_params

    def postprocess(
        self,
        model_outputs: Any,
        function_to_apply: Any = None,
        top_k: int | None = None,
        gen_threshold: float | None = None,
        char_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Convert logits into caption/tag dictionaries."""
        logits = model_outputs["logits"][0]
        if self.framework == "pt":
            logits = logits.to(torch.float32)

        probs = torch.sigmoid(logits)
        effective_gen = gen_threshold if gen_threshold is not None else self._default_gen_threshold
        effective_char = (
            char_threshold if char_threshold is not None else self._default_char_threshold
        )

        caption, taglist, ratings, character, general = get_tags(
            probs=probs,
            labels=self._label_data,
            gen_threshold=effective_gen,
            char_threshold=effective_char,
        )

        result = {
            "caption": caption,
            "tag_string": taglist,
            "ratings": ratings,
            "character": character,
            "general": general,
        }
        return [result]


def _create_warning_callback() -> Callable[[str], None]:
    def emit(message: str) -> None:
        if message.startswith("Warning:"):
            logger.warning(message)
        else:
            logger.info(message)

    return emit


def wd_tagger_pipeline(
    model: str = "eva02-large",
    *,
    repo_id: str | None = None,
    adapter: str | None = None,
    labels_path: str | Path | None = None,
    revision: str | None = None,
    adapter_revision: str | None = None,
    token: str | None = None,
    adapter_token: str | None = None,
    gen_threshold: float = 0.35,
    char_threshold: float = 0.75,
    quantization: bool = False,
    precision: str | None = None,
    device_map: str | None = "auto",
    **kwargs: Any,
) -> WDTaggerPipeline:
    """Factory that loads a WD Tagger model and returns a ready-to-use pipeline."""
    model_key = model or "eva02-large"
    base_identifier = _resolve_base_model_identifier(model_key, repo_id)
    warning_callback = _create_warning_callback()

    adapter_token_final = adapter_token or token
    peft_config: PeftConfig | None = None
    if adapter is not None:
        logger.info("Loading adapter configuration from '%s'", adapter)
        peft_config = PeftConfig.from_pretrained(
            adapter,
            revision=adapter_revision,
            token=adapter_token_final,
        )
        adapter_base = cast("str", peft_config.base_model_name_or_path)
        if repo_id is None and model_key in MODEL_REPO_MAP:
            base_identifier = adapter_base
        logger.info(
            "Adapter trained on base model '%s'. Using '%s' for inference.",
            adapter_base,
            base_identifier,
        )

    labels_path_resolved = Path(labels_path).expanduser() if labels_path is not None else None
    labels: LabelData = _load_label_data(
        base_repo=base_identifier,
        revision=revision,
        token=token,
        labels_path=labels_path_resolved,
        adapter=adapter,
        adapter_revision=adapter_revision,
        adapter_token=adapter_token_final,
        fallback_repo=(peft_config.base_model_name_or_path if peft_config is not None else None),
        warning_callback=warning_callback,
    )

    try:
        precision_dtype = _parse_precision(precision)
    except ValueError as exc:
        msg = f"Invalid precision value: {exc}"
        raise ValueError(msg) from exc

    quantization_config = _create_quantization_config() if quantization else None

    base_kwargs: dict[str, Any] = {
        "num_labels": len(labels.names),
        "ignore_mismatched_sizes": True,
    }
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

    logger.info("Loading base model from '%s' (%d labels)...", base_identifier, len(labels.names))
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
        logger.info("Applying adapter weights from '%s'...", adapter)
        hf_model = PeftModel.from_pretrained(hf_model, adapter, **adapter_kwargs)

    hf_model = hf_model.eval()

    logger.info("Initializing image processor...")
    image_processor = WDTaggerImageProcessor(pretrained_model_name_or_path=base_identifier)

    return WDTaggerPipeline(
        model=hf_model,
        image_processor=image_processor,
        label_data=labels,
        gen_threshold=gen_threshold,
        char_threshold=char_threshold,
        **kwargs,
    )


PIPELINE_REGISTRY.register_pipeline(
    task="wd-tagger",
    pipeline_class=WDTaggerPipeline,
    pt_model=AutoModelForImageClassification,
    type="image-classification",
    default={
        "model": AutoModelForImageClassification,
        "image_processor": WDTaggerImageProcessor,
    },
)


__all__ = ["WDTaggerPipeline", "wd_tagger_pipeline"]
