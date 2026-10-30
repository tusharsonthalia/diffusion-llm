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

@triton.jit
def _dkdv_tiles(
    k_tile, v_tile,      # this key tile
    kv_block,            # its keys' block ids
    q_desc, dO_desc,
    L_ptr, d_ptr,        # L, Delta of this (batch, head)
    stride_ls,
    dK, dV,
    lo, hi,              # QUERY range [lo, hi)
    n, block_size, sm_scale, shift,
    BLOCK_Q: tl.constexpr,
    RULE: tl.constexpr,
):
    for start_m in range(lo, hi, BLOCK_Q):
        q_tile = q_desc.load([start_m, 0])
        dO_tile = dO_desc.load([start_m, 0])

        q_idx = start_m + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
        l_row = tl.load(L_ptr + q_idx * stride_ls)
        d_row = tl.load(d_ptr + q_idx * stride_ls)

        s_t = tl.dot(k_tile, tl.trans(q_tile)) * sm_scale  # [BLOCK_KV, BLOCK_Q]
        p_t = tl.exp(s_t - l_row[None, :])
        if RULE != 0:
            q_block = (q_idx % n) // block_size
            if RULE == 1:  # noisy keys: noisy queries in the same block
                mask = kv_block[:, None] == q_block[None, :]
            if RULE == 2:  # clean keys: later (shift 1) / same or later (0)
                mask = q_block[None, :] >= kv_block[:, None] + shift
            p_t = tl.where(mask, p_t, 0.0)  # type: ignore

        dV = tl.dot(p_t.to(dO_tile.dtype), dO_tile, acc=dV)
        dP_t = tl.dot(v_tile, tl.trans(dO_tile))
        dS_t = p_t * (dP_t - d_row[None, :])
        dK = tl.dot(dS_t.to(q_tile.dtype), q_tile, acc=dK)

    return dK, dV

@triton.jit
def block_sparse_attention_dk_dv_bwd_kern(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dK_ptr, dV_ptr, L_ptr, Delta_ptr,
    stride_b, stride_h, stride_s,     # q, k, v, dO, dK, dV
    stride_lb, stride_lh, stride_ls,  # L, Delta
    n_heads, n, block_size, sm_scale,
    D: tl.constexpr, BLOCK_KV: tl.constexpr, BLOCK_Q: tl.constexpr,
):
    batch_index = tl.program_id(0) // n_heads
    head_index = tl.program_id(0) % n_heads
    k_start = tl.program_id(1) * BLOCK_KV
    base = batch_index * stride_b + head_index * stride_h

    q_desc = tl.make_tensor_descriptor(
        base=Q_ptr + base,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_Q, D],
    )
    dO_desc = tl.make_tensor_descriptor(
        base=dO_ptr + base,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_Q, D],
    )
    k_desc = tl.make_tensor_descriptor(
        base=K_ptr + base,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_KV, D],
    )
    v_desc = tl.make_tensor_descriptor(
        base=V_ptr + base,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_KV, D],
    )
    dK_desc = tl.make_tensor_descriptor(
        base=dK_ptr + base,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_KV, D],
    )
    dV_desc = tl.make_tensor_descriptor(
        base=dV_ptr + base,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_KV, D],
    )

    k_tile = k_desc.load([k_start, 0])
    v_tile = v_desc.load([k_start, 0])
    kv_idx = k_start + tl.arange(0, BLOCK_KV)  # [BLOCK_KV]
    kv_block = (kv_idx % n) // block_size

    base_l = batch_index * stride_lb + head_index * stride_lh
    l_ptr = L_ptr + base_l
    d_ptr = Delta_ptr + base_l

    dK = tl.zeros([BLOCK_KV, D], dtype=tl.float32)
    dV = tl.zeros([BLOCK_KV, D], dtype=tl.float32)

    c0 = k_start                 # first key
    c1 = k_start + BLOCK_KV - 1  # last key

    if k_start < n:  # noisy keys: noisy queries in the same block
        # first key's block start (rounded down to a query tile) to the
        # last key's block end; clamped to n, noisy queries only
        lo = ((c0 // block_size) * block_size // BLOCK_Q) * BLOCK_Q
        hi = tl.minimum((c1 // block_size + 1) * block_size, n)
        dK, dV = _dkdv_tiles(
            k_tile, v_tile, kv_block, q_desc, dO_desc, l_ptr, d_ptr,
            stride_ls, dK, dV, lo, hi, n, block_size, sm_scale, 0,
            BLOCK_Q, 1,
        )
    else:  # clean keys
        kb_first = (c0 - n) // block_size  # first key's clean block
        kb_last = (c1 - n) // block_size   # last key's clean block

        # noisy queries in later blocks: partial from the block after the
        # first key's, fully visible from the block after the last key's
        lo = tl.minimum(
            ((kb_first + 1) * block_size // BLOCK_Q) * BLOCK_Q, n
        )
        full = tl.minimum(
            tl.cdiv((kb_last + 1) * block_size, BLOCK_Q) * BLOCK_Q, n
        )
        dK, dV = _dkdv_tiles(
            k_tile, v_tile, kv_block, q_desc, dO_desc, l_ptr, d_ptr,
            stride_ls, dK, dV, lo, full, n, block_size, sm_scale, 1,
            BLOCK_Q, 2,
        )
        dK, dV = _dkdv_tiles(
            k_tile, v_tile, kv_block, q_desc, dO_desc, l_ptr, d_ptr,
            stride_ls, dK, dV, full, n, n, block_size, sm_scale, 1,
            BLOCK_Q, 0,
        )

        # clean queries in the same or later blocks: partial from the first
        # key's block, fully visible from the last key's block
        lo = n + (kb_first * block_size // BLOCK_Q) * BLOCK_Q
        full = n + tl.minimum(
            tl.cdiv(kb_last * block_size, BLOCK_Q) * BLOCK_Q, n
        )
        dK, dV = _dkdv_tiles(
            k_tile, v_tile, kv_block, q_desc, dO_desc, l_ptr, d_ptr,
            stride_ls, dK, dV, lo, full, n, block_size, sm_scale, 0,
            BLOCK_Q, 2,
        )
        dK, dV = _dkdv_tiles(
            k_tile, v_tile, kv_block, q_desc, dO_desc, l_ptr, d_ptr,
            stride_ls, dK, dV, full, 2 * n, n, block_size, sm_scale, 0,
            BLOCK_Q, 0,
        )

    dK_desc.store([k_start, 0], (dK * sm_scale).to(dK_ptr.dtype.element_ty))
    dV_desc.store([k_start, 0], dV.to(dV_ptr.dtype.element_ty))

@triton.jit
def _dq_tiles(
    q_tile, dO_tile,     # this query tile and its dO
    l_row, d_row,        # its rows' L and Delta
    q_block,             # its rows' block ids
    k_desc, v_desc,
    dQ,
    lo, hi,              # key range [lo, hi)
    n, block_size, sm_scale, shift,
    BLOCK_N: tl.constexpr,
    RULE: tl.constexpr,  # same 0 / 1 / 2 as the forward
):
    for start_n in range(lo, hi, BLOCK_N):
        k_tile = k_desc.load([start_n, 0])
        v_tile = v_desc.load([start_n, 0])

        s_tile = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        p_tile = tl.exp(s_tile - l_row[:, None])

        kv_idx = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        if RULE == 1:  # noisy keys, block diagonal
            kv_block = kv_idx // block_size
            mask = q_block[:, None] == kv_block[None, :]
        if RULE == 2:  # clean keys, (offset) block-causal
            kv_block = (kv_idx - n) // block_size
            mask = q_block[:, None] >= kv_block[None, :] + shift
        if RULE != 0:
            p_tile = tl.where(mask, p_tile, 0.0) # type: ignore

        dP = tl.dot(dO_tile, tl.trans(v_tile))

        dS = p_tile * (dP - d_row[:, None])

        dQ = tl.dot(dS.to(k_tile.dtype), k_tile, acc=dQ)

    return dQ


@triton.jit
def block_sparse_attention_bwd_dq_kern(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dQ_ptr, L_ptr, Delta_ptr,
    stride_b, stride_h, stride_s,     # q, k, v, dO, dQ
    stride_lb, stride_lh, stride_ls,  # L, Delta
    n_heads, n, block_size, sm_scale,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    batch_index = tl.program_id(0) // n_heads
    head_index =  tl.program_id(0) % n_heads
    q_start = tl.program_id(1) * BLOCK_M

    q_desc = tl.make_tensor_descriptor(
        base=Q_ptr + batch_index * stride_b + head_index * stride_h,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_M, D],
    )
    k_desc = tl.make_tensor_descriptor(
        base=K_ptr + batch_index * stride_b + head_index * stride_h,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_N, D],
    )
    v_desc = tl.make_tensor_descriptor(
        base=V_ptr + batch_index * stride_b + head_index * stride_h,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_N, D],
    )
    dO_desc = tl.make_tensor_descriptor(
        base=dO_ptr + batch_index * stride_b + head_index * stride_h,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_M, D],
    )
    dQ_desc = tl.make_tensor_descriptor(
        base=dQ_ptr + batch_index * stride_b + head_index * stride_h,
        shape=[2 * n, D],
        strides=[stride_s, 1],
        block_shape=[BLOCK_M, D],
    )
    q_tile = q_desc.load([q_start, 0])
    dO_tile = dO_desc.load([q_start, 0])
    q_idx = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]

    base_l = batch_index * stride_lb + head_index * stride_lh
    l_row = tl.load(L_ptr + base_l + q_idx * stride_ls)
    d_row = tl.load(Delta_ptr + base_l + q_idx * stride_ls)
    dQ = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # same key ranges as the forward
    r0 = q_start                # first row
    r1 = q_start + BLOCK_M - 1  # last row (sees the most keys)
    q_block = (q_idx % n) // block_size  # block id within each copy
    shift = (q_start < n).to(tl.int32)   # noisy: earlier blocks only

    if q_start < n:  # noisy tile
        # own noisy blocks; clamped to n so no key is counted twice
        lo_a = ((r0 // block_size) * block_size // BLOCK_N) * BLOCK_N
        hi_a = tl.minimum((r1 // block_size + 1) * block_size, n)
        dQ = _dq_tiles(
            q_tile, dO_tile, l_row, d_row, q_block, k_desc, v_desc, dQ,
            lo_a, hi_a, n, block_size, sm_scale, shift, BLOCK_N, 1,
        )
        # earlier clean blocks; full: those before the first row's block
        full = n + ((r0 // block_size) * block_size // BLOCK_N) * BLOCK_N
        hi = n + (r1 // block_size) * block_size
    else:  # clean tile: own and earlier clean blocks
        # full: up to the end of the first row's block
        full = n + (((r0 - n) // block_size + 1) * block_size
                    // BLOCK_N) * BLOCK_N
        hi = tl.minimum(n + ((r1 - n) // block_size + 1) * block_size, 2 * n)

    # clean keys: [n, full) visible to every row, [full, hi) partial
    dQ = _dq_tiles(
        q_tile, dO_tile, l_row, d_row, q_block, k_desc, v_desc, dQ,
        n, full, n, block_size, sm_scale, shift, BLOCK_N, 0,
    )
    dQ = _dq_tiles(
        q_tile, dO_tile, l_row, d_row, q_block, k_desc, v_desc, dQ,
        full, hi, n, block_size, sm_scale, shift, BLOCK_N, 2,
    )

    dQ_desc.store([q_start, 0], (dQ * sm_scale).to(dQ_ptr.dtype.element_ty))

class BlockSparseFlashAttention(torch.autograd.Function):
    """Attention under the BD3-LM block diffusion mask."""

    # forward config from autotuning on an A100 (best at L >= 4096)
    FWD_BLOCK_M, FWD_BLOCK_N = 128, 64
    FWD_NUM_WARPS, FWD_NUM_STAGES = 4, 3
    BWD_DQ_BLOCK_M, BWD_DQ_BLOCK_N = 128, 64
    BWD_DQ_NUM_WARPS, BWD_DQ_NUM_STAGES = 4, 3
    BWD_DKDV_BLOCK_KV, BWD_DKDV_BLOCK_Q = 128, 32
    BWD_DKDV_NUM_WARPS, BWD_DKDV_NUM_STAGES = 4, 2

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
    ) -> tuple[
        Float[Tensor, "n_heads d_model head_dim"],
        Float[Tensor, "n_heads d_model head_dim"],
        Float[Tensor, "n_heads d_model head_dim"],
        None
    ]:
        q, k, v, O, L = ctx.saved_tensors
        block_size = ctx.block_size
        B, H, seq_len, D = q.shape
        n = seq_len // 2
        assert n % block_size == 0

        dO = dO.contiguous()
        delta = (dO.float() * O.float()).sum(-1)
        dQ, dK, dV = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        sm_scale = 1 / D ** 0.5

        assert dO.stride() == q.stride() == k.stride() == dQ.stride()
        assert delta.stride() == L.stride()

        dq_grid = (B * H, seq_len // BlockSparseFlashAttention.BWD_DQ_BLOCK_M)
        block_sparse_attention_bwd_dq_kern[dq_grid]( # type: ignore
            q, k, v,
            dO, dQ, L, delta,
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            H, n, block_size, sm_scale, D,
            BlockSparseFlashAttention.BWD_DQ_BLOCK_M,
            BlockSparseFlashAttention.BWD_DQ_BLOCK_N,
            num_warps=BlockSparseFlashAttention.BWD_DQ_NUM_WARPS,
            num_stages=BlockSparseFlashAttention.BWD_DQ_NUM_STAGES,
        )

        BLOCK_KV = BlockSparseFlashAttention.BWD_DKDV_BLOCK_KV
        BLOCK_Q = BlockSparseFlashAttention.BWD_DKDV_BLOCK_Q

        assert n % BLOCK_KV == 0 and BLOCK_KV % BLOCK_Q == 0

        dkdv_grid = (B * H, seq_len // BLOCK_KV)  # one program per key tile
        block_sparse_attention_dk_dv_bwd_kern[dkdv_grid]( # type: ignore
            q, k, v,
            dO, dK, dV, L, delta,
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            H, n, block_size, sm_scale,
            D=D, BLOCK_KV=BLOCK_KV, BLOCK_Q=BLOCK_Q,
            num_warps=BlockSparseFlashAttention.BWD_DKDV_NUM_WARPS,
            num_stages=BlockSparseFlashAttention.BWD_DKDV_NUM_STAGES,
        )

        return dQ, dK, dV, None