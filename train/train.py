import argparse
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor, nn

from bench.harness import log_result
from model.bd3lm import BD3LM
from model.mdlm import MDLM
from train.bd3lm_loss import bd3lm_loss
from train.data import draw_random_batch
from train.mdlm_loss import mdlm_loss
from train.noise import LogLinearNoise

# GPT-2's 50257 tokens + [MASK] (matches meta.json's model_vocab_size)
DEFAULT_VOCAB_SIZE = 50258


@dataclass
class TrainConfig:
    model: str = "bd3lm"                # "mdlm" | "bd3lm"
    # data: dir with train.bin (scripts/setup.sh)
    data_dir: str = "/vol/data/openwebtext"
    seq_len: int = 1024
    batch_size: int = 16
    # model (reference "small": 768 / 12 / 12 / 3072 / 128)
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    d_cond: int = 128
    block_size: int = 16                # BD3-LM only
    # BD3-LM clipped schedule [beta, omega]
    sampling_eps_min: float = 1e-3
    sampling_eps_max: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # run
    steps: int = 100
    warmup_steps: int = 10              # excluded from timing
    sync_debug: bool = False            # warn on GPU->CPU syncs in the loop
    amp_dtype: str = "bf16"             # "bf16" | "fp32"
    seed: int = 0


def build_model(cfg: TrainConfig, vocab_size: int) -> nn.Module:
    """MDLM or BD3-LM with the configured DiT backbone."""
    common = {
        "vocab_size": vocab_size,
        "max_seq_len": cfg.seq_len,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "d_ff": cfg.d_ff,
        "d_cond": cfg.d_cond,
        "device": torch.device("cuda"),
    }
    if cfg.model == "mdlm":
        return MDLM(**common)
    if cfg.model == "bd3lm":
        return BD3LM(**common, block_size=cfg.block_size)
    raise ValueError(f"unknown model {cfg.model!r}")


def compute_loss(
    cfg: TrainConfig,
    model: nn.Module,
    noise: LogLinearNoise,
    x0: Int[Tensor, "batch seq"],
) -> Tensor:
    """The diffusion loss for one batch of clean tokens x0 (a scalar)."""
    if cfg.model == "mdlm":
        return mdlm_loss(model, noise, x0)
    if cfg.model == "bd3lm":
        return bd3lm_loss(
            model,
            x0,
            cfg.sampling_eps_min,
            cfg.sampling_eps_max,
        )
    raise ValueError(f"unknown model {cfg.model!r}")


def train_step(
    cfg: TrainConfig,
    model: nn.Module,
    noise: LogLinearNoise,
    optimizer: torch.optim.Optimizer,
    next_batch: Callable[[], np.ndarray],
    amp_dtype: torch.dtype | None,
) -> tuple[Tensor, Tensor]:
    """One complete training step.

    Returns (loss, pre-clip grad norm) as 0-d DEVICE tensors.
    """
    # pinned + non_blocking memory transfer
    with torch.cuda.nvtx.range("data"):
        x0 = torch.from_numpy(next_batch())
        x0 = x0.pin_memory().to("cuda", non_blocking=True)

    with torch.cuda.nvtx.range("forward"), torch.autocast(
        device_type="cuda",
        dtype=amp_dtype or torch.bfloat16,
        enabled=amp_dtype is not None,
    ):
        loss = compute_loss(cfg, model, noise, x0)

    with torch.cuda.nvtx.range("backward"):
        loss.backward()

    with torch.cuda.nvtx.range("optimizer"):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return loss.detach(), grad_norm


def main(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    amp_dtype = {"bf16": torch.bfloat16, "fp32": None}[cfg.amp_dtype]

    np.random.seed(cfg.seed)
    device = torch.device("cuda")

    # data: the tokenized train split, memory-mapped (never loaded into RAM)
    vocab_size = DEFAULT_VOCAB_SIZE
    tokens = np.memmap(
        Path(cfg.data_dir) / "train.bin", dtype=np.uint16, mode="r"
    )

    def next_batch() -> np.ndarray:
        return draw_random_batch(tokens, cfg.batch_size, cfg.seq_len)

    # model, schedule, optimizer
    model = build_model(cfg, vocab_size).train()
    noise = LogLinearNoise()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.model}: {n_params / 1e6:.1f}M params | {asdict(cfg)}")

    torch.cuda.reset_peak_memory_stats(device)

    losses = torch.zeros(cfg.steps, device=device)
    grad_norms = torch.zeros(cfg.steps, device=device)

    start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(cfg.steps)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(cfg.steps)
    ]

    if cfg.sync_debug:
        torch.cuda.set_sync_debug_mode("warn")

    for step in range(cfg.steps):
        start_events[step].record()

        with torch.cuda.nvtx.range(f"step {step}"):
            loss, grad_norm = train_step(
                cfg, model, noise, optimizer, next_batch, amp_dtype
            )

        # device-to-device writes: no sync
        losses[step] = loss
        grad_norms[step] = grad_norm

        end_events[step].record()

    torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize(device)  # the run's only sync
    step_ms = [
        s.elapsed_time(e)
        for s, e in zip(start_events, end_events, strict=True)
    ]

    # one device -> host copy of the whole run
    losses_host = losses.tolist()
    grad_norms_host = grad_norms.tolist()

    timed_ms = step_ms[cfg.warmup_steps:]
    if not timed_ms:
        print("no timed steps (steps <= warmup_steps)")
        return
    median_ms = statistics.median(timed_ms)
    tokens_per_s = cfg.batch_size * cfg.seq_len / (median_ms / 1e3)
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    print(
        f"loss {losses_host[0]:.4f} -> {losses_host[-1]:.4f} | "
        f"median step {median_ms:.1f} ms | "
        f"{tokens_per_s:,.0f} tokens/s | peak mem {peak_mem_gb} GB"
    )

    log_result(
        "train_step",
        {**asdict(cfg), "n_params": n_params},
        median_step_ms=median_ms,
        step_ms=step_ms,
        tokens_per_s=tokens_per_s,
        peak_mem_gb=peak_mem_gb,
        losses=losses_host,
        grad_norms=grad_norms_host,
    )


def parse_args() -> TrainConfig:
    """Every TrainConfig field becomes a --flag (bools are switches)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    main(parse_args())
