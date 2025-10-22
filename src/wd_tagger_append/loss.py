from typing import Any, Literal

import torch
from timm.loss import asymmetric_loss
from torch import Tensor
from transformers.modeling_outputs import ImageClassifierOutput


class AsymmetricLossMultiLabel(asymmetric_loss.AsymmetricLossMultiLabel):
    """Adapter that supports Trainer-style outputs, gradient scaling, and class weighting."""

    def __init__(
        self,
        *args,
        reduction: Literal["sum", "mean"] = "sum",
        class_weights: Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reduction = reduction
        self.register_buffer("_class_weights", None, persistent=False)
        if class_weights is not None:
            self.update_class_weights(class_weights)

    def update_class_weights(self, weights: Tensor | None) -> None:
        """Store class weights as a buffer for device placement consistency."""
        if weights is None:
            self._class_weights = None
            return

        if weights.ndim != 1:
            msg = "class_weights must be a 1D tensor"
            raise ValueError(msg)
        self._class_weights = weights.detach().clone().to(dtype=torch.float32)

    def forward(self, x: ImageClassifierOutput, y: Tensor, **kwargs: Any) -> Tensor:
        num_items_in_batch = kwargs.get("num_items_in_batch")

        logits = x.logits
        loss_matrix = self._compute_loss_matrix(logits, y)

        if self._class_weights is not None:
            weights = self._class_weights.to(loss_matrix.device)
            loss_matrix = loss_matrix * weights.view(1, -1)

        loss = loss_matrix.mean() if self._reduction == "mean" else loss_matrix.sum()

        if num_items_in_batch is not None:
            if isinstance(num_items_in_batch, int):
                loss = loss / num_items_in_batch
            else:
                loss = loss / num_items_in_batch.to(loss.device)

        return loss

    def _compute_loss_matrix(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Return per-example, per-class loss prior to reduction."""
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = labels * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - labels) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * labels
            pt1 = xs_neg * (1 - labels)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * labels + self.gamma_neg * (1 - labels)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss = loss * one_sided_w

        return -loss
