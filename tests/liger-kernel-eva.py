import copy
from collections.abc import Iterable

import torch
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.transformers import LigerLayerNorm
from timm.layers.mlp import SwiGLU
from torch import nn
from transformers import AutoModelForImageClassification

# -----------------------------------------------------------------------------
# Helpers: LayerNorm
# -----------------------------------------------------------------------------


def _extract_normalized_shape_size(
    normalized_shape: int | Iterable[int] | torch.Size,
) -> int:
    """Return the flattened hidden size from LayerNorm.normalized_shape."""
    if isinstance(normalized_shape, int):
        return normalized_shape
    result = 1
    for d in normalized_shape:
        result *= int(d)
    return int(result)


def _extract_device_and_dtype(
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> tuple[torch.device, torch.dtype]:
    """Extract a device/dtype hint from weight or bias, default to CPU/FP32."""
    if weight is not None:
        return weight.device, weight.dtype
    if bias is not None:
        return bias.device, bias.dtype
    return torch.device("cpu"), torch.float32


def _copy_parameter_with_grad(
    destination: nn.Parameter,
    source: torch.Tensor,
    requires_grad: bool,
) -> None:
    with torch.no_grad():
        destination.copy_(source.detach().reshape_as(destination))
    destination.requires_grad = requires_grad


def _initialize_parameter(parameter: nn.Parameter, value: float, requires_grad: bool) -> None:
    with torch.no_grad():
        parameter.fill_(value)
    parameter.requires_grad = requires_grad


def _create_liger_layer_norm(
    hidden_size: int,
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> LigerLayerNorm:
    ln = LigerLayerNorm(hidden_size=hidden_size, eps=eps, bias=True)
    return ln.to(device=device, dtype=dtype)


def _transfer_layernorm_weight(
    destination: LigerLayerNorm,
    source: nn.LayerNorm,
    has_affine: bool,
) -> None:
    if has_affine and getattr(source, "weight", None) is not None:  # type: ignore[attr-defined]
        _copy_parameter_with_grad(destination.weight, source.weight, source.weight.requires_grad)  # type: ignore[arg-type]
    else:
        _initialize_parameter(destination.weight, 1.0, False)  # type: ignore[arg-type]


def _transfer_layernorm_bias(
    destination: LigerLayerNorm,
    source: nn.LayerNorm,
    has_affine: bool,
) -> None:
    if has_affine and getattr(source, "bias", None) is not None:  # type: ignore[attr-defined]
        _copy_parameter_with_grad(destination.bias, source.bias, source.bias.requires_grad)  # type: ignore[arg-type]
    else:
        _initialize_parameter(destination.bias, 0.0, False)  # type: ignore[arg-type]


def make_liger_from_layernorm(ln: nn.LayerNorm) -> LigerLayerNorm:
    """Convert a single nn.LayerNorm (or subclass) to LigerLayerNorm.

    Preserves weights, bias, eps, device/dtype and requires_grad.
    If ln.elementwise_affine is False, initializes weight=1, bias=0 and freezes them.
    """
    hidden_size = _extract_normalized_shape_size(getattr(ln, "normalized_shape", ()))
    eps = float(getattr(ln, "eps", 1e-6))
    has_affine = bool(getattr(ln, "elementwise_affine", True))
    device, dtype = _extract_device_and_dtype(
        getattr(ln, "weight", None),
        getattr(ln, "bias", None),
    )

    dst = _create_liger_layer_norm(hidden_size, eps, device, dtype)
    _transfer_layernorm_weight(dst, ln, has_affine)
    _transfer_layernorm_bias(dst, ln, has_affine)
    return dst


def convert_layernorms_to_liger(module: nn.Module, *, inplace: bool = True) -> nn.Module:
    """Replace all nn.LayerNorm (and subclasses) with LigerLayerNorm recursively."""
    root = module if inplace else copy.deepcopy(module)

    def _walk(m: nn.Module) -> None:
        for name, child in list(m.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(m, name, make_liger_from_layernorm(child))
            else:
                _walk(child)

    _walk(root)
    return root


# -----------------------------------------------------------------------------
# SwiGLU -> LigerSwiGLUCompat
# -----------------------------------------------------------------------------


class LigerSwiGLUCompat(nn.Module):
    """A SwiGLU block using LigerSiLUMulFunction, matching reference semantics:
    x_gate = fc1_g(x); x = fc1_x(x); x = SiLU(x_gate) * x; drop1; norm; fc2; drop2.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        bias_g: bool = True,
        bias_x: bool = True,
        bias_out: bool = True,
        norm: nn.Module | None = None,
        drop1: float = 0.0,
        drop2: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias_g, **factory)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=bias_x, **factory)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=bias_out, **factory)

        self.drop1 = nn.Dropout(drop1) if drop1 and drop1 > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop2) if drop2 and drop2 > 0 else nn.Identity()
        self.norm = norm if norm is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = LigerSiLUMulFunction.apply(g, u)
        h = self.drop1(h)
        h = self.norm(h)
        h = self.down_proj(h)
        return self.drop2(h)


def _drop_p(mod: nn.Module) -> float:
    return getattr(mod, "p", 0.0) if isinstance(mod, nn.Dropout) else 0.0


def make_liger_swiglu_from_swiglu(src: SwiGLU) -> LigerSwiGLUCompat:
    """Port weights, bias, dropout, and norm from a timm SwiGLU block."""
    in_features = src.fc1_g.in_features
    hidden_features = src.fc1_g.out_features
    out_features = src.fc2.out_features

    bias_g = src.fc1_g.bias is not None
    bias_x = src.fc1_x.bias is not None
    bias_out = src.fc2.bias is not None

    device, dtype = src.fc1_g.weight.device, src.fc1_g.weight.dtype

    dst = LigerSwiGLUCompat(
        in_features,
        hidden_features,
        out_features,
        bias_g=bias_g,
        bias_x=bias_x,
        bias_out=bias_out,
        norm=copy.deepcopy(src.norm),
        drop1=_drop_p(src.drop1),
        drop2=_drop_p(src.drop2),
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        dst.gate_proj.weight.copy_(src.fc1_g.weight.detach())
        dst.up_proj.weight.copy_(src.fc1_x.weight.detach())
        dst.down_proj.weight.copy_(src.fc2.weight.detach())
        dst.gate_proj.weight.requires_grad = src.fc1_g.weight.requires_grad
        dst.up_proj.weight.requires_grad = src.fc1_x.weight.requires_grad
        dst.down_proj.weight.requires_grad = src.fc2.weight.requires_grad
        if bias_g:
            dst.gate_proj.bias.copy_(src.fc1_g.bias.detach())  # type: ignore[arg-type]
            dst.gate_proj.bias.requires_grad = src.fc1_g.bias.requires_grad  # type: ignore[union-attr]
        if bias_x:
            dst.up_proj.bias.copy_(src.fc1_x.bias.detach())  # type: ignore[arg-type]
            dst.up_proj.bias.requires_grad = src.fc1_x.bias.requires_grad  # type: ignore[union-attr]
        if bias_out:
            dst.down_proj.bias.copy_(src.fc2.bias.detach())  # type: ignore[arg-type]
            dst.down_proj.bias.requires_grad = src.fc2.bias.requires_grad  # type: ignore[union-attr]

    return dst


def convert_swiglu_to_liger(module: nn.Module, *, inplace: bool = True) -> nn.Module:
    """Recursively replace timm SwiGLU with LigerSwiGLUCompat."""
    root = module if inplace else copy.deepcopy(module)

    def _walk(m: nn.Module) -> None:
        for name, child in list(m.named_children()):
            if isinstance(child, SwiGLU):
                setattr(m, name, make_liger_swiglu_from_swiglu(child))
            else:
                _walk(child)

    _walk(root)
    return root


# -----------------------------------------------------------------------------
# End-to-end helpers
# -----------------------------------------------------------------------------


def convert_model_to_liger(module: nn.Module, *, inplace: bool = True) -> nn.Module:
    """Convert a model by swapping LayerNorm and SwiGLU to Liger variants."""
    root = module if inplace else copy.deepcopy(module)
    root = convert_layernorms_to_liger(root, inplace=True)
    return convert_swiglu_to_liger(root, inplace=True)


# -----------------------------------------------------------------------------
# Demo / self-check
# -----------------------------------------------------------------------------


def _self_check() -> None:
    print("Loading model...")
    model = AutoModelForImageClassification.from_pretrained("SmilingWolf/wd-eva02-large-tagger-v3")
    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Creating test input...")
    x = torch.randn(1, 3, 448, 448, device=device)

    print("Running original model...")
    with torch.no_grad():
        out_ref = model(x)

    print("Converting to Liger...")
    model_liger = convert_model_to_liger(model, inplace=False).to(device)

    print("Running converted model...")
    with torch.no_grad():
        out_new = model_liger(x)

    print("Comparing logits...")
    is_close = torch.allclose(out_ref.logits, out_new.logits, atol=1e-5)
    diff = torch.abs(out_ref.logits - out_new.logits)
    print(f"Outputs match (atol=1e-5): {is_close}")
    print(f"Max diff: {diff.max().item():.3e}")
    print(f"Mean diff: {diff.mean().item():.3e}")
    print(f"Std  diff: {diff.std().item():.3e}")


if __name__ == "__main__":
    _self_check()
