import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from bench.attention import BACKENDS
from bench.harness import bench
from kernels.BlockSparseFlashAttention import BlockSparseFlashAttention
from model.bd3lm import block_diff_mask

H, D = 12, 64
CUDA = torch.device("cuda")
TOKENS = 16384
CONFIGS = [(n, 16) for n in (1024, 2048, 4096, 8192)] + [
    (4096, bs) for bs in (4, 64, 256)
]
flex = torch.compile(flex_attention, dynamic=False)
torch._functorch.config.donated_buffer = False


def bd3lm_mask_mod(seq_len, block_size):
    # block_diff_mask for a single (query, key) index pair; FlexAttention
    # passes batch and head too, which the mask doesn't depend on
    def mask_mod(batch, head, q_idx, kv_idx):
        q_clean, kv_clean = q_idx >= seq_len, kv_idx >= seq_len
        q_block = torch.where(q_clean, q_idx - seq_len, q_idx) // block_size
        kv_block = torch.where(kv_clean, kv_idx - seq_len, kv_idx) // block_size

        block_diagonal = ~q_clean & ~kv_clean & (q_block == kv_block)
        offset_block_causal = ~q_clean & kv_clean & (kv_block < q_block)
        block_causal = q_clean & kv_clean & (kv_block <= q_block)
        return block_diagonal | offset_block_causal | block_causal

    return mask_mod


def backends(n, block_size):
    mask = block_diff_mask(n, block_size, CUDA)
    block_mask = create_block_mask(
        bd3lm_mask_mod(n, block_size), None, None, 2 * n, 2 * n, CUDA
    )
    return mask, {
        "custom": lambda q, k, v: BACKENDS["custom"](q, k, v, mask),
        "sdpa": lambda q, k, v: BACKENDS["sdpa"](q, k, v, mask),
        "flex": lambda q, k, v: flex(q, k, v, block_mask=block_mask),
        "bsfa": lambda q, k, v: BlockSparseFlashAttention.apply(
            q, k, v, block_size),
    }


def randn_qkv(batch, seq_len, dtype):
    shape = (batch, H, seq_len, D)
    return [torch.randn(shape, dtype=dtype, device=CUDA, requires_grad=True)
            for _ in range(3)]


def time_fwd_bwd(fn, q, k, v, warmup=5, rep=20):
    dout = torch.randn_like(q)

    def grad(out):
        return torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)

    fwd = bench(lambda: fn(q, k, v), warmup, rep).median_ms
    out = fn(q, k, v)
    bwd = bench(lambda: grad(out), warmup, rep).median_ms
    del out

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    grad(fn(q, k, v))
    peak_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
    return fwd, bwd, peak_gb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="")
    out_dir = p.parse_args().out_dir

    rows = []
    for n, block_size in CONFIGS:
        L, B = 2 * n, max(1, TOKENS // (2 * n))
        mask, fns = backends(n, block_size)
        q, k, v = randn_qkv(B, L, torch.bfloat16)
        fwd_flops = 4 * B * H * L**2 * D  # QK^T + PV, backward is 2.5x

        for name, fn in fns.items():
            row = {"backend": name, "L": L, "batch": B,
                   "block_size": block_size,
                   "density": mask.float().mean().item()}
            try:
                with torch.autocast("cuda", torch.bfloat16):
                    fwd_ms, bwd_ms, peak_gb = time_fwd_bwd(fn, q, k, v)
                fwd_tflops = fwd_flops / fwd_ms / 1e9
                bwd_tflops = 2.5 * fwd_flops / bwd_ms / 1e9
                row["fwd_ms"] = fwd_ms
                row["bwd_ms"] = bwd_ms
                row["peak_gb"] = peak_gb
                row["fwd_tflops"] = fwd_tflops
                row["bwd_tflops"] = bwd_tflops
                print(f"{name:10s} L={L:5d} B={B} block={block_size:3d} "
                      f"fwd {fwd_ms:7.2f} ms bwd {bwd_ms:7.2f} ms "
                      f"peak {peak_gb:5.1f} GB "
                      f"{fwd_tflops:5.1f}/{bwd_tflops:5.1f} TFLOPS")
            except torch.OutOfMemoryError:
                row["oom"] = True
                print(f"{name:10s} L={L:5d} B={B} OOM")
            # free what an OOM left behind before the next backend runs
            gc.collect()
            torch.cuda.empty_cache()
            rows.append(row)

    if out_dir:
        result = {"gpu": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "rows": rows}
        out = Path(out_dir) / "results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
