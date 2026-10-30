"""Block-sparse FlashAttention-2 Triton kernels for the BD3-LM training mask."""

from typing import Any

import torch
from torch import Tensor
from jaxtyping import Float
import triton # type: ignore
import triton.language as tl # type: ignore


@triton.jit
def bd3lm_mask(
    q_idx,   # [BLOCK_M, 1]
    kv_idx,  # [1, BLOCK_N]
    n,
    block_size,
):
    """block_diff_mask for one tile: True where the query may attend."""
    q_clean = q_idx >= n
    kv_clean = kv_idx >= n

    q_block_idx = (q_idx % n) // block_size
    kv_block_idx = (kv_idx % n) // block_size

    mask = (~q_clean & ~kv_clean) & (q_block_idx == kv_block_idx)
    mask |= (~q_clean & kv_clean) & (q_block_idx > kv_block_idx)
    mask |= (q_clean & kv_clean) & (q_block_idx >= kv_block_idx)

    return mask # [BLOCK_M, BLOCK_N]

@triton.jit
def _fwd_tiles(
    q_tile, q_block,     # this query tile and each row's block id
    k_desc, v_desc,      # K, V of this (batch, head)
    m, l, output,        # running softmax state
    lo, hi,              # key range [lo, hi)
    n, block_size, qk_scale,
    shift,               # rule 2: 1 for noisy rows (<), 0 for clean (<=)
    BLOCK_N: tl.constexpr,
    RULE: tl.constexpr,  # 0 no mask, 1 same block, 2 earlier clean blocks
):
    """Online softmax over the key tiles in [lo, hi); returns the updated
    (m, l, output)."""
    for start_n in range(lo, hi, BLOCK_N):
        k_tile = k_desc.load([start_n, 0])
        v_tile = v_desc.load([start_n, 0])

        # [BLOCK_M, BLOCK_N]
        s_tile = tl.dot(q_tile, tl.trans(k_tile)) * qk_scale

        # one comparison per element
        kv_idx = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        if RULE == 1:  # noisy keys, block diagonal
            kv_block = kv_idx // block_size
            mask = q_block[:, None] == kv_block[None, :]
        if RULE == 2:  # clean keys, (offset) block-causal
            kv_block = (kv_idx - n) // block_size
            mask = q_block[:, None] >= kv_block[None, :] + shift
        if RULE != 0:
            s_tile = tl.where(mask, s_tile, -1e6) # type: ignore

        m_new = tl.maximum(m, tl.max(s_tile, axis=1))
        p_tile = tl.exp(s_tile - m_new[:, None])
        alpha = tl.exp(m - m_new)

        l = alpha * l + tl.sum(p_tile, axis=1)
        output = alpha[:, None] * output
        output = tl.dot(p_tile.to(v_tile.dtype), v_tile, acc=output)
        m = m_new

    return m, l, output


@triton.jit
def block_sparse_attention_fwd_kern(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    n_heads, n,
    block_size, qk_scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Forward pass: one program per (batch * head, query tile).

    Visits only the key tiles the query tile can see, masking only the
    partial ones. Writes O and L, the per-row logsumexp in natural log.
    """
    # heads on the fast grid axis: programs running together get equal work
    batch_index = tl.program_id(0) // n_heads
    head_index = tl.program_id(0) % n_heads
    q_start = tl.program_id(1) * BLOCK_M

    q_desc = tl.make_tensor_descriptor(
        base=Q_ptr + batch_index * stride_qb + head_index * stride_qh,
        shape=[2 * n, D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D],
    )
    k_desc = tl.make_tensor_descriptor(
        base=K_ptr + batch_index * stride_kb + head_index * stride_kh,
        shape=[2 * n, D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D],
    )
    v_desc = tl.make_tensor_descriptor(
        base=V_ptr + batch_index * stride_vb + head_index * stride_vh,
        shape=[2 * n, D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D],
    )
    o_desc = tl.make_tensor_descriptor(
        base=O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=[2 * n, D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, D],
    )

    q_tile = q_desc.load([q_start, 0])
    q_idx = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]

    output = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    # tile skipping: loop only over the key ranges this tile can see
    r0 = q_start                # first row
    r1 = q_start + BLOCK_M - 1  # last row (sees the most keys)
    q_block = (q_idx % n) // block_size  # block id within each copy
    shift = (q_start < n).to(tl.int32)   # noisy: earlier blocks only

    if q_start < n:  # noisy tile
        # own noisy blocks; clamped to n so no key is counted twice
        lo_a = ((r0 // block_size) * block_size // BLOCK_N) * BLOCK_N
        hi_a = tl.minimum((r1 // block_size + 1) * block_size, n)
        m, l, output = _fwd_tiles(
            q_tile, q_block, k_desc, v_desc, m, l, output, lo_a, hi_a,
            n, block_size, qk_scale, shift, BLOCK_N, 1,
        )
        # earlier clean blocks; full: those before the first row's block
        full = n + ((r0 // block_size) * block_size // BLOCK_N) * BLOCK_N
        hi = n + (r1 // block_size) * block_size
    else:  # clean tile: own and earlier clean blocks
        # full: up to the end of the first row's block
        full = n + (((r0 - n) // block_size + 1) * block_size
                    // BLOCK_N) * BLOCK_N
        hi = tl.minimum(n + ((r1 - n) // block_size + 1) * block_size, 2 * n)

    # clean keys, shared by both tile kinds: [n, full) is visible to every
    # row (no mask), [full, hi) is partial
    m, l, output = _fwd_tiles(
        q_tile, q_block, k_desc, v_desc, m, l, output, n, full,
        n, block_size, qk_scale, shift, BLOCK_N, 0,
    )
    m, l, output = _fwd_tiles(
        q_tile, q_block, k_desc, v_desc, m, l, output, full, hi,
        n, block_size, qk_scale, shift, BLOCK_N, 2,
    )

    output = (output / l[:, None]).to(O_ptr.dtype.element_ty)
    o_desc.store([q_start, 0], output)

    lse = m + tl.log(l)
    l_ptrs = (L_ptr + batch_index * stride_lb + head_index * stride_lh
              + q_idx * stride_ls)
    tl.store(l_ptrs, lse, mask=q_idx < 2 * n)


class BlockSparseFlashAttention(torch.autograd.Function):
    """Attention under the BD3-LM block diffusion mask."""

    # forward config from autotuning on an A100 (best at L >= 4096)
    FWD_BLOCK_M, FWD_BLOCK_N = 128, 64
    FWD_NUM_WARPS, FWD_NUM_STAGES = 4, 3

    @staticmethod
    def forward(
        ctx: Any,
        q: Float[Tensor, "batch heads seq_len head_dim"],
        k: Float[Tensor, "batch heads seq_len head_dim"],
        v: Float[Tensor, "batch heads seq_len head_dim"],
        block_size: int,
    ) -> Float[Tensor, "batch heads seq_len head_dim"]:
        B, H, seq_len, D = q.shape
        assert q.is_cuda, "inputs must be on CUDA"
        assert q.shape == k.shape == v.shape, "q, k, v must match"
        assert seq_len % 2 == 0, "BD3-LM input is cat(x_t, x_0): length 2n"
        n = seq_len // 2

        # tensor descriptors need contiguous rows (last stride 1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        O = torch.empty_like(q)
        L = torch.empty((B, H, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M = BlockSparseFlashAttention.FWD_BLOCK_M
        BLOCK_N = BlockSparseFlashAttention.FWD_BLOCK_N
        # a query tile is all noisy or all clean; n falls on a key-tile edge
        assert n % BLOCK_M == 0 and BLOCK_M % BLOCK_N == 0
        grid = (B * H, triton.cdiv(seq_len, BLOCK_M))

        block_sparse_attention_fwd_kern[grid]( # type: ignore
            q, k, v, O, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            H, n,
            block_size, 1 / D ** 0.5,
            D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=BlockSparseFlashAttention.FWD_NUM_WARPS,
            num_stages=BlockSparseFlashAttention.FWD_NUM_STAGES,
        )

        ctx.save_for_backward(q, k, v, O, L)
        ctx.block_size = block_size
        return O

    @staticmethod
    def backward( # type: ignore
        ctx: Any,
        dO: Float[Tensor, "batch heads seq_len head_dim"],
    ):
        raise NotImplementedError("Backward kernel missing")