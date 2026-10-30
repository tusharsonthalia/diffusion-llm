"""Split-KV (FlashDecoding) attention for BD3-LM cached sampling."""

import torch
import triton
import triton.language as tl
from torch import Tensor

from kernels.BlockSparseDecodeAttention import (
    _decode_tiles, _last_stride_1, block_sparse_decode_attention,
)

# keys per split: 3584 keys x 12 heads -> 14 x 12 = 168 programs
SPLIT_N = 256


@triton.jit
def split_kv_kern(
    Q_ptr, K_ptr, V_ptr, O_part_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    n_heads, n_splits, q_len, kv_len,
    block_size, qk_scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_N: tl.constexpr,
):
    """One program per (batch * head, split): online softmax over the
    split's SPLIT_N keys, written out as a normalized partial + its LSE."""
    bh = tl.program_id(0)
    split = tl.program_id(1)
    batch_index = bh // n_heads
    head_index = bh % n_heads

    q_desc = tl.make_tensor_descriptor(
        base=Q_ptr + batch_index * stride_qb + head_index * stride_qh,
        shape=[q_len, D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D],
    )
    k_desc = tl.make_tensor_descriptor(
        base=K_ptr + batch_index * stride_kb + head_index * stride_kh,
        shape=[kv_len, D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D],
    )
    v_desc = tl.make_tensor_descriptor(
        base=V_ptr + batch_index * stride_vb + head_index * stride_vh,
        shape=[kv_len, D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D],
    )

    # all queries fit in one tile, the last q_len key positions
    q_tile = q_desc.load([0, 0])
    q_offset = kv_len - q_len
    q_block = (q_offset + tl.arange(0, BLOCK_M)) // block_size

    # keys before `full` are seen by every row; the rest need the mask
    full = tl.minimum((q_offset // block_size + 1) * block_size, kv_len)
    full = (full // BLOCK_N) * BLOCK_N

    lo = split * SPLIT_N
    hi = tl.minimum(lo + SPLIT_N, kv_len)
    mid = tl.minimum(tl.maximum(full, lo), hi)

    output = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    m, l, output = _decode_tiles(
        q_tile, q_block, k_desc, v_desc, m, l, output, lo, mid,
        kv_len, block_size, qk_scale, BLOCK_N, False,
    )
    m, l, output = _decode_tiles(
        q_tile, q_block, k_desc, v_desc, m, l, output, mid, hi,
        kv_len, block_size, qk_scale, BLOCK_N, True,
    )

    # partials laid out (batch * heads, n_splits, BLOCK_M, [D])
    slot = (bh * n_splits + split) * BLOCK_M
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    output = output / l[:, None]
    tl.store(O_part_ptr + (slot + rows[:, None]) * D + cols[None, :], output)
    tl.store(LSE_ptr + slot + rows, m + tl.log(l))


@triton.jit
def split_kv_reduce_kern(
    O_part_ptr, LSE_ptr, O_ptr,
    stride_ob, stride_oh, stride_os,
    n_heads, n_splits, q_len,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Merge one (batch, head)'s splits, weighting each partial by
    exp(LSE_s - max LSE)."""
    bh = tl.program_id(0)
    batch_index = bh // n_heads
    head_index = bh % n_heads
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)

    output = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    weight = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    for split in range(n_splits):
        slot = (bh * n_splits + split) * BLOCK_M
        lse = tl.load(LSE_ptr + slot + rows)
        part = tl.load(O_part_ptr + (slot + rows[:, None]) * D + cols[None, :])

        m_new = tl.maximum(m, lse)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(lse - m_new)
        output = alpha[:, None] * output + beta[:, None] * part
        weight = alpha * weight + beta
        m = m_new

    o_desc = tl.make_tensor_descriptor(
        base=O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=[q_len, D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, D],
    )
    o_desc.store([0, 0], (output / weight[:, None]).to(O_ptr.dtype.element_ty))


def split_kv_decode_attention(
    q: Tensor,  # (batch, heads, q_len, head_dim): the last q_len positions
    k: Tensor,  # (batch, heads, kv_len, head_dim)
    v: Tensor,
    block_size: int,
) -> Tensor:
    """Same result as block_sparse_decode_attention, with the keys split
    across programs so a decode step fills the GPU."""
    B, H, q_len, D = q.shape
    kv_len = k.shape[2]
    if q_len > 64:
        # prefill: many query tiles already fill the GPU
        return block_sparse_decode_attention(q, k, v, block_size)
    q, k, v = _last_stride_1(q), _last_stride_1(k), _last_stride_1(v)

    BLOCK_M = 16 if q_len <= 16 else 64
    BLOCK_N = 64
    n_splits = triton.cdiv(kv_len, SPLIT_N)
    o_part = torch.empty(B * H, n_splits, BLOCK_M, D, device=q.device,
                         dtype=torch.float32)
    lse = torch.empty(B * H, n_splits, BLOCK_M, device=q.device,
                      dtype=torch.float32)
    O = torch.empty_like(q)

    split_kv_kern[(B * H, n_splits)](  # type: ignore
        q, k, v, o_part, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        H, n_splits, q_len, kv_len,
        block_size, 1 / D ** 0.5,
        D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, SPLIT_N=SPLIT_N,
        num_warps=4, num_stages=2,
    )
    split_kv_reduce_kern[(B * H,)](  # type: ignore
        o_part, lse, O,
        O.stride(0), O.stride(1), O.stride(2),
        H, n_splits, q_len,
        D=D, BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return O
