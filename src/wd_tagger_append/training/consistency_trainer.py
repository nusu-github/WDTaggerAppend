"""Custom Trainer that injects a teacher-consistency loss term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Trainer


@dataclass(slots=True)
class ConsistencyConfig:
    """Configuration for the consistency trainer."""

    weight: float
    warmup_ratio: float
    teacher: nn.Module | None


class ConsistencyTrainer(Trainer):
    """Trainer with an additional KL divergence penalty between student and teacher logits."""

    def __init__(self, *, consistency: ConsistencyConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.consistency = consistency
        self.teacher = consistency.teacher
        self._teacher_device: torch.device | None = None
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad_(False)

    def _get_base_loss(
        self,
        outputs: Any,
        logits: Tensor | None,
        labels: Tensor | None,
    ) -> Tensor:
        if outputs.loss is not None:
            return outputs.loss
        if logits is not None and labels is not None:
            return F.binary_cross_entropy_with_logits(logits, labels)
        device = logits.device if logits is not None else self.args.device
        return torch.zeros((), device=device)

    def _consistency_scale(self) -> float:
        ratio = max(self.consistency.warmup_ratio, 0.0)
        if ratio <= 0.0:
            return 1.0
        max_steps = self.state.max_steps or self.args.max_steps or 0
        if max_steps <= 0:
            return 1.0
        warmup_steps = max(1, int(max_steps * ratio))
        return min(1.0, self.state.global_step / warmup_steps)

    def _maybe_move_teacher(self, student_logits: Tensor) -> None:
        if self.teacher is None:
            return
        target_device = student_logits.device
        if self._teacher_device != target_device:
            self.teacher = self.teacher.to(device=target_device, dtype=student_logits.dtype)
            self._teacher_device = target_device

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        logits: Tensor | None = getattr(outputs, "logits", None)
        labels = inputs.get("labels")
        base_loss = self._get_base_loss(outputs, logits, labels)

        teacher = self.teacher
        if (
            teacher is None
            or logits is None
            or self.consistency.weight <= 0.0
            or not torch.isfinite(base_loss)
        ):
            if return_outputs:
                return base_loss, outputs
            return base_loss

        self._maybe_move_teacher(logits)
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        with torch.no_grad():
            teacher_outputs = teacher(**teacher_inputs)
        teacher_logits: Tensor | None = getattr(teacher_outputs, "logits", None)
        if teacher_logits is None:
            if return_outputs:
                return base_loss, outputs
            return base_loss

        student_prob = torch.sigmoid(logits)
        teacher_prob = torch.sigmoid(teacher_logits.to(student_prob.dtype))
        eps = torch.finfo(student_prob.dtype).eps
        student_log_prob = torch.log(student_prob.clamp_min(eps))

        kl_term = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        if not torch.isfinite(kl_term):
            if return_outputs:
                return base_loss, outputs
            return base_loss

        scale = self._consistency_scale()
        total_loss = base_loss + scale * self.consistency.weight * kl_term

        if torch.isfinite(kl_term):
            scaled_penalty = scale * self.consistency.weight * kl_term.detach()
            self.log({"loss_consistency": float(scaled_penalty.item())})

        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs
        return total_loss
