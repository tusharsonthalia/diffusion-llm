from collections.abc import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from model import modules

AttentionFn = Callable[..., Tensor]

FORWARD_RANGE = "attention"
BACKWARD_RANGE = "attention backward"


def torch_sdpa(
    q: Float[Tensor, "... seq_q d_k"],
    k: Float[Tensor, "... seq_k d_k"],
    v: Float[Tensor, "... seq_k d_v"],
    mask: Bool[Tensor, "... seq_q seq_k"] | None = None,
) -> Float[Tensor, "... seq_q d_v"]:
    """PyTorch SDPA. With mask=None (MDLM) it can use the FlashAttention
    backend; with BD3-LM's boolean block mask it falls back to the
    memory-efficient backend (mask as an additive bias, no tile skipping).
    """
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


BACKENDS: dict[str, AttentionFn] = {
    "custom": modules.scaled_dot_product_attention,
    "sdpa": torch_sdpa,
}


class _OpenRange(torch.autograd.Function):
    """Identity on (q, k, v) that opens the forward NVTX range."""
    @staticmethod
    def forward(ctx, q, k, v):
        torch.cuda.nvtx.range_push(FORWARD_RANGE)
        return q.view_as(q), k.view_as(k), v.view_as(v)

    @staticmethod
    def backward(ctx, *grad_outputs):  # (dq, dk, dv)
        torch.cuda.nvtx.range_pop()
        return grad_outputs


class _CloseRange(torch.autograd.Function):
    """Identity on the attention output that closes the forward range."""
    @staticmethod
    def forward(ctx, o):
        torch.cuda.nvtx.range_pop()
        return o.view_as(o)

    @staticmethod
    def backward(ctx, *grad_outputs):  # (do,)
        torch.cuda.nvtx.range_push(BACKWARD_RANGE)
        return grad_outputs


def with_nvtx(attention: AttentionFn) -> AttentionFn:
    """Wrap a backend in NVTX ranges for both forward and backward."""

    def annotated(q, k, v, mask=None):
        q, k, v = _OpenRange.apply(q, k, v)
        return _CloseRange.apply(attention(q, k, v, mask))

    return annotated


def install_attention(name: str, nvtx: bool = False) -> None:
    """Route every attention layer through the named backend."""
    attention = BACKENDS[name]
    modules.scaled_dot_product_attention = (
        with_nvtx(attention) if nvtx else attention
    )
