from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import TrainingArguments
from transformers.modeling_outputs import ImageClassifierOutput

from wd_tagger_append.training import ConsistencyConfig, ConsistencyTrainer


class DummyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ImageClassifierOutput:
        logits = self.linear(pixel_values)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        return ImageClassifierOutput(logits=logits, loss=loss)


def make_trainer(tmp_path, weight: float, warmup_ratio: float, teacher: nn.Module | None):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=1,
        remove_unused_columns=False,
        report_to=[],
        no_cuda=True,
    )
    model = DummyClassifier()
    config = ConsistencyConfig(weight=weight, warmup_ratio=warmup_ratio, teacher=teacher)
    return model, ConsistencyTrainer(
        consistency=config,
        model=model,
        args=args,
    )


def test_consistency_penalty_matches_expected(tmp_path) -> None:
    student, trainer = make_trainer(
        tmp_path,
        weight=0.5,
        warmup_ratio=0.5,
        teacher=DummyClassifier(),
    )
    trainer.consistency.teacher.load_state_dict(student.state_dict())

    trainer.state.max_steps = 10
    trainer.state.global_step = 2

    inputs = {
        "pixel_values": torch.randn(2, 4),
        "labels": torch.rand(2, 3),
    }

    student_outputs = student(**inputs)
    expected_base = student_outputs.loss
    teacher_outputs = trainer.consistency.teacher(pixel_values=inputs["pixel_values"])
    student_prob = torch.sigmoid(student_outputs.logits)
    teacher_prob = torch.sigmoid(teacher_outputs.logits)

    eps = torch.finfo(student_prob.dtype).eps
    kl_term = F.kl_div(torch.log(student_prob.clamp_min(eps)), teacher_prob, reduction="batchmean")
    warmup_scale = trainer.state.global_step / 5  # 10 * 0.5 warmup steps
    expected_total = expected_base + trainer.consistency.weight * warmup_scale * kl_term

    total_loss = trainer.compute_loss(student, {k: v.clone() for k, v in inputs.items()})
    assert torch.allclose(total_loss, expected_total, atol=1e-6)


def test_zero_weight_skips_teacher(tmp_path) -> None:
    student, trainer = make_trainer(tmp_path, weight=0.0, warmup_ratio=0.0, teacher=None)
    inputs = {
        "pixel_values": torch.randn(2, 4),
        "labels": torch.rand(2, 3),
    }

    base_loss = student(**inputs).loss
    total_loss = trainer.compute_loss(student, {k: v.clone() for k, v in inputs.items()})
    assert torch.allclose(total_loss, base_loss, atol=1e-6)
