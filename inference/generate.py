import argparse
import time

import torch

from bench.attention import install_attention
from model.bd3lm import BD3LM
from sample.sampler import sample_bd3lm
from train.train import VOCAB_SIZE

BLOCK_SIZE = 16
PROMPT_LEN = 3072
GEN_LEN = 1024


def build_model():
    # the "small" training config
    return BD3LM(VOCAB_SIZE, PROMPT_LEN + GEN_LEN, 768, 12, 12, 3072, 128,
                 BLOCK_SIZE, device=torch.device("cuda")).eval()


def baseline():
    """No KV cache: every step re-encodes the whole sequence. fp32 weights
    under autocast, eager."""
    install_attention("bsdfa", nvtx=True, block_size=BLOCK_SIZE)
    return build_model(), False


def bf16_weights():
    """+ matmul weights stored in bf16, so autocast stops recasting them
    every forward."""
    model, kv_cache = baseline()
    for p in model.parameters():
        if p.ndim == 2:
            p.data = p.data.to(torch.bfloat16)
    return model, kv_cache


def compiled():
    """+ torch.compile, dynamic: the sequence grows every block."""
    model, kv_cache = bf16_weights()
    # attention NVTX ranges would graph-break compile
    install_attention("bsdfa", block_size=BLOCK_SIZE)
    torch._dynamo.config.recompile_limit = 64
    model.backbone.compile(dynamic=True)
    return model, kv_cache


def kv_cached():
    """+ KV cache: the prompt is prefilled once, then each step runs only
    the current block."""
    model, _ = compiled()
    return model, True


def split_kv():
    """+ split-KV attention: a decode step's keys are spread over many
    programs instead of one per head."""
    model, kv_cache = kv_cached()
    install_attention("split_kv", block_size=BLOCK_SIZE)
    return model, kv_cache


METHODS = {
    "baseline": baseline,
    "bf16": bf16_weights,
    "compile": compiled,
    "kv_cache": kv_cached,
    "split_kv": split_kv,
}


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    args = parser.parse_args()

    torch.manual_seed(0)
    model, kv_cache = METHODS[args.method]()
    prompt = torch.randint(0, VOCAB_SIZE - 1, (1, PROMPT_LEN), device="cuda")

    def generate(n_blocks):
        sample_bd3lm(model, 1, n_blocks, BLOCK_SIZE, kv_cache, prompt)

    with torch.autocast("cuda", torch.bfloat16):
        generate(2)  # warmup (and compile)
        first_block = timed(lambda: generate(1))
        torch.cuda.cudart().cudaProfilerStart()
        total = timed(lambda: generate(GEN_LEN // BLOCK_SIZE))
        torch.cuda.cudart().cudaProfilerStop()

    print(f"{args.method} on {torch.cuda.get_device_name()}")
    print(f"prompt {PROMPT_LEN}, generated {GEN_LEN}, block {BLOCK_SIZE}")
    print(f"time to first block {first_block * 1000:.1f} ms")
    print(f"{GEN_LEN / total:.1f} tokens/s ({total:.2f} s)")


if __name__ == "__main__":
    main()
