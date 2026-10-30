from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from kernels.BlockSparseDecodeAttention import block_sparse_decode_attention
from kernels.BlockSparseFlashAttention import BlockSparseFlashAttention
from kernels.SplitKVDecodeAttention import split_kv_decode_attention
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


_flex = torch.compile(flex_attention)
_block_masks = {}


def flex(q, k, v, mask=None):
    if mask is None:
        return _flex(q, k, v)
    key = (mask.data_ptr(), tuple(mask.shape))
    if key not in _block_masks:
        def mask_mod(batch, head, q_idx, kv_idx):
            return mask[q_idx, kv_idx]

        _block_masks[key] = create_block_mask(
            mask_mod, None, None, *mask.shape, device=mask.device)
    return _flex(q, k, v, block_mask=_block_masks[key])


def bsfa(q, k, v, mask=None, block_size=None):
    """Block-sparse FlashAttention (forward + backward) on the 2n training
    layout; it builds the mask from block_size."""
    return BlockSparseFlashAttention.apply(q, k, v, block_size)


def bsdfa(q, k, v, mask=None, block_size=None):
    """Block-sparse decode attention (forward only): the last q_len
    positions, block-causal; it builds the mask from block_size."""
    return block_sparse_decode_attention(q, k, v, block_size)


def split_kv(q, k, v, mask=None, block_size=None):
    """Split-KV decode attention (forward only): bsdfa with the keys split
    across programs, for decode steps with few queries."""
    return split_kv_decode_attention(q, k, v, block_size)


BACKENDS: dict[str, AttentionFn] = {
    "custom": modules.scaled_dot_product_attention,
    "sdpa": torch_sdpa,
    "flex": flex,
    "bsfa": bsfa,
    "bsdfa": bsdfa,
    "split_kv": split_kv,
}
# backends that ignore the mask and need the block size instead
BLOCK_SIZE_BACKENDS = ("bsfa", "bsdfa", "split_kv")


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


def install_attention(
    name: str, nvtx: bool = False, block_size: int | None = None
) -> None:
    """Route every attention layer through the named backend."""
    attention = BACKENDS[name]
    if name in BLOCK_SIZE_BACKENDS:
        assert block_size is not None, f"{name} needs block_size"
        attention = partial(attention, block_size=block_size)
    modules.scaled_dot_product_attention = (
        with_nvtx(attention) if nvtx else attention
    )
