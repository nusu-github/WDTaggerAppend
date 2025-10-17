import torch
from timm.loss import asymmetric_loss
from transformers.modeling_outputs import ImageClassifierOutput


class AsymmetricLossMultiLabel(asymmetric_loss.AsymmetricLossMultiLabel):
    def forward(self, x: ImageClassifierOutput, targets: torch.Tensor, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        # Extract num_items_in_batch if present (used for gradient accumulation)
        num_items_in_batch = kwargs.get("num_items_in_batch")

        # Compute base loss
        loss = super().forward(x.logits, targets)

        # If num_items_in_batch is provided, the loss should already be a sum reduction
        # and we need to divide by num_items_in_batch for proper averaging across accumulation steps
        if num_items_in_batch is not None:
            if isinstance(num_items_in_batch, int):
                loss = loss / num_items_in_batch
            else:
                # num_items_in_batch is a tensor
                loss = loss / num_items_in_batch.to(loss.device)

        return loss
