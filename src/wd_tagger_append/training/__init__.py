"""Training extensions for consistency-regularised fine-tuning."""

from .consistency_trainer import ConsistencyConfig, ConsistencyTrainer
from .teacher import build_teacher_model

__all__ = ["ConsistencyConfig", "ConsistencyTrainer", "build_teacher_model"]
