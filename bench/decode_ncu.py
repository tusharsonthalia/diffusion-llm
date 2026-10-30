"""One sampling step's attention, for Nsight Compute: batch 1, 12 heads,
the shapes of the inference ladder's 3072 + 1024 run.

    cached:   a 16-token block over a 3584-token cache (grid 12 x 1)
    uncached: the whole 3088-token sequence, block-causal (grid 12 x 49)

    python -m bench.decode_ncu <cached | uncached> <bsdfa | split_kv>
"""
import sys

import torch

from kernels.BlockSparseDecodeAttention import block_sparse_decode_attention
from kernels.SplitKVDecodeAttention import split_kv_decode_attention

BLOCK_SIZE = 16
SHAPES = {"cached": (BLOCK_SIZE, 3584), "uncached": (3088, 3088)}
KERNELS = {"bsdfa": block_sparse_decode_attention,
           "split_kv": split_kv_decode_attention}


def main():
    q_len, kv_len = SHAPES[sys.argv[1]]
    attention = KERNELS[sys.argv[2]]
    q = torch.randn(1, 12, q_len, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 12, kv_len, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    for _ in range(3):  # the first call compiles; ncu profiles the third
        attention(q, k, v, BLOCK_SIZE)


if __name__ == "__main__":
    main()
