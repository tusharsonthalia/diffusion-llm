"""Block-sparse FlashAttention for BD3-LM sampling."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _decode_tiles(
    q_tile, q_block, k_desc, v_desc, m, l, output,
    lo, hi, kv_len, block_size, qk_scale,
    BLOCK_N: tl.constexpr,
    MASK: tl.constexpr,
):
    """Online softmax over key tiles [lo, hi); MASK applies the
    block-causal rule and the key bound."""
    for start_n in range(lo, hi, BLOCK_N):
        k_tile = k_desc.load([start_n, 0])
        v_tile = v_desc.load([start_n, 0])
        s_tile = tl.dot(q_tile, tl.trans(k_tile)) * qk_scale

        if MASK:
            kv_idx = start_n + tl.arange(0, BLOCK_N)
            mask = q_block[:, None] >= (kv_idx // block_size)[None, :]
            mask &= kv_idx[None, :] < kv_len
            s_tile = tl.where(mask, s_tile, -1e6)

        m_new = tl.maximum(m, tl.max(s_tile, axis=1))
        p_tile = tl.exp(s_tile - m_new[:, None])
        alpha = tl.exp(m - m_new)

        l = alpha * l + tl.sum(p_tile, axis=1)
        output = alpha[:, None] * output
        output = tl.dot(p_tile.to(v_tile.dtype), v_tile, acc=output)
        m = m_new
    return m, l, output


@triton.jit
def block_sparse_decode_kern(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    n_heads, q_len, kv_len,
    block_size, qk_scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_index = tl.program_id(0) // n_heads
    head_index = tl.program_id(0) % n_heads
    q_start = tl.program_id(1) * BLOCK_M

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
    o_desc = tl.make_tensor_descriptor(
        base=O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=[q_len, D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, D],
    )

    q_tile = q_desc.load([q_start, 0])

    q_offset = kv_len - q_len
    q_block = (q_offset + q_start + tl.arange(0, BLOCK_M)) // block_size
    first_block = (q_offset + q_start) // block_size
    last_row = tl.minimum(q_start + BLOCK_M, q_len) - 1
    last_block = (q_offset + last_row) // block_size

    # tile skipping: [0, full) seen by every row (no mask), [full, hi)
    # partial, keys past hi seen by no row
    full = tl.minimum((first_block + 1) * block_size, kv_len)
    full = (full // BLOCK_N) * BLOCK_N
    hi = tl.minimum((last_block + 1) * block_size, kv_len)

    output = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    m, l, output = _decode_tiles(
        q_tile, q_block, k_desc, v_desc, m, l, output, 0, full,
        kv_len, block_size, qk_scale, BLOCK_N, False,
    )
    m, l, output = _decode_tiles(
        q_tile, q_block, k_desc, v_desc, m, l, output, full, hi,
        kv_len, block_size, qk_scale, BLOCK_N, True,
    )

    output = (output / l[:, None]).to(O_ptr.dtype.element_ty)
    o_desc.store([q_start, 0], output)


def _last_stride_1(x: Tensor) -> Tensor:
    # tensor descriptors need contiguous rows; other strides are fine
    return x if x.stride(-1) == 1 else x.contiguous()


def block_sparse_decode_attention(
    q: Tensor,  # (batch, heads, q_len, head_dim): the last q_len positions
    k: Tensor,  # (batch, heads, kv_len, head_dim)
    v: Tensor,
    block_size: int,
) -> Tensor:
    """Block-causal attention of the last q_len positions over all kv_len
    keys; no KV splits: one program per (batch * head, query tile)."""
    B, H, q_len, D = q.shape
    kv_len = k.shape[2]
    assert q_len <= kv_len, "queries are the last q_len key positions"
    q, k, v = _last_stride_1(q), _last_stride_1(k), _last_stride_1(v)
    O = torch.empty_like(q)

    # a cached step has only block_size queries; tl.dot needs >= 16 rows
    BLOCK_M = 16 if q_len <= 16 else 64
    BLOCK_N = 64
    grid = (B * H, triton.cdiv(q_len, BLOCK_M))

    block_sparse_decode_kern[grid](  # type: ignore
        q, k, v, O,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        H, q_len, kv_len,
        block_size, 1 / D ** 0.5,
        D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return O
