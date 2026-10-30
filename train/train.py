import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor, nn

from bench.attention import BACKENDS, install_attention
from bench.harness import log_result
from model.bd3lm import BD3LM
from model.mdlm import MDLM
from train.bd3lm_loss import bd3lm_loss
from train.data import draw_random_batch
from train.mdlm_loss import mdlm_loss
from train.noise import LogLinearNoise

# GPT-2's 50257 tokens + [MASK] (matches meta.json's model_vocab_size)
VOCAB_SIZE = 50258

StepFn = Callable[..., tuple[Tensor, Tensor]]


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
    # optimizations, one flag each
    step: str = "baseline"              # "baseline" | "overlap"
    attn: str = "custom"                # "custom" | "sdpa"
    fused_adam: bool = False
    compile: bool = False
    # NVTX ranges around attention fwd + bwd (nsys; eager only)
    nvtx_attn: bool = False
    # run
    steps: int = 100
    warmup_steps: int = 10              # excluded from timing
    profile_steps: int = 0              # >0: cudaProfiler range for nsys
    sync_debug: bool = False            # warn on GPU->CPU syncs in the loop
    amp_dtype: str = "bf16"             # "bf16" | "fp32"
    seed: int = 0
    out_dir: str = ""                   # write config.json + metrics.json


def build_model(cfg: TrainConfig) -> nn.Module:
    """MDLM or BD3-LM with the configured DiT backbone."""
    common = {
        "vocab_size": VOCAB_SIZE,
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
    x0: Int[Tensor, "batch seq"],
    amp_dtype: torch.dtype | None,
) -> tuple[Tensor, Tensor]:
    """Forward, backward and optimizer update on a batch already on the GPU.

    Shared by every step variant: they differ only in how x0 gets there.
    Returns (loss, pre-clip grad norm) as 0-d device tensors.
    """
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


def to_gpu(batch: np.ndarray) -> Tensor:
    """Host batch -> GPU without a sync: pinned memory + non_blocking."""
    return torch.from_numpy(batch).pin_memory().to("cuda", non_blocking=True)


def baseline_step(
    cfg: TrainConfig,
    model: nn.Module,
    noise: LogLinearNoise,
    optimizer: torch.optim.Optimizer,
    next_batch: Callable[[], np.ndarray],
    amp_dtype: torch.dtype | None,
) -> tuple[Tensor, Tensor]:
    """Fetch this step's batch, copy it on the compute stream, train."""
    with torch.cuda.nvtx.range("data"):
        x0 = to_gpu(next_batch())

    return train_step(cfg, model, noise, optimizer, x0, amp_dtype)


class OverlapStep:
    """Prefetch: copy step N+1's batch on a side stream during step N.

    The compute stream only waits for a copy issued one step earlier, which
    has normally finished by then, so the transfer is off the critical path.
    """

    def __init__(self):
        self.copy_stream = torch.cuda.Stream()
        self.next_x0: Tensor | None = None

    def _prefetch(self, next_batch: Callable[[], np.ndarray]) -> None:
        with torch.cuda.nvtx.range("prefetch"):
            host = next_batch()
            with torch.cuda.stream(self.copy_stream):
                self.next_x0 = to_gpu(host)

    def __call__(
        self,
        cfg: TrainConfig,
        model: nn.Module,
        noise: LogLinearNoise,
        optimizer: torch.optim.Optimizer,
        next_batch: Callable[[], np.ndarray],
        amp_dtype: torch.dtype | None,
    ) -> tuple[Tensor, Tensor]:
        if self.next_x0 is None:
            self._prefetch(next_batch)

        # this step's batch: make compute wait for its copy
        compute = torch.cuda.current_stream()
        compute.wait_stream(self.copy_stream)
        x0 = self.next_x0
        assert x0 is not None
        x0.record_stream(compute)

        # start the next batch's copy; it runs while this step computes
        self._prefetch(next_batch)

        return train_step(cfg, model, noise, optimizer, x0, amp_dtype)


def make_step(name: str) -> StepFn:
    """The train-step variant selected by --step."""
    if name == "baseline":
        return baseline_step
    if name == "overlap":
        return OverlapStep()
    raise ValueError(f"unknown step {name!r}")


def main(cfg: TrainConfig) -> None:
    assert cfg.steps > cfg.warmup_steps + cfg.profile_steps, (
        "steps must exceed warmup_steps + profile_steps"
    )
    assert cfg.attn in BACKENDS, f"unknown attention backend {cfg.attn!r}"
    assert not (cfg.nvtx_attn and cfg.compile), (
        "nvtx_attn graph-breaks torch.compile"
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda")
    amp_dtype = {"bf16": torch.bfloat16, "fp32": None}[cfg.amp_dtype]

    # data: the tokenized train split, memory-mapped (never loaded into RAM)
    tokens = np.memmap(
        Path(cfg.data_dir) / "train.bin", dtype=np.uint16, mode="r"
    )

    def next_batch() -> np.ndarray:
        return draw_random_batch(tokens, cfg.batch_size, cfg.seq_len)

    # attention backend first
    install_attention(cfg.attn, nvtx=cfg.nvtx_attn)

    model = build_model(cfg).train()
    n_params = sum(p.numel() for p in model.parameters())
    runner = torch.compile(model) if cfg.compile else model
    noise = LogLinearNoise()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
        fused=cfg.fused_adam,
    )
    train_step_fn = make_step(cfg.step)
    print(f"{cfg.model}: {n_params / 1e6:.1f}M params | {asdict(cfg)}")


    losses = torch.zeros(cfg.steps, device=device)
    grad_norms = torch.zeros(cfg.steps, device=device)

    start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(cfg.steps)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(cfg.steps)
    ]

    # nsys capture window: steady-state steps right after warmup
    profile_start = cfg.warmup_steps
    profile_end = cfg.warmup_steps + cfg.profile_steps

    if cfg.sync_debug:
        torch.cuda.set_sync_debug_mode("warn")

    for step in range(cfg.steps):
        if cfg.profile_steps and step == profile_start:
            torch.cuda.profiler.start()

        start_events[step].record()
        with torch.cuda.nvtx.range(f"step {step}"):
            loss, grad_norm = train_step_fn(
                cfg, runner, noise, optimizer, next_batch, amp_dtype
            )
        end_events[step].record()

        # device-to-device writes: no sync
        losses[step] = loss
        grad_norms[step] = grad_norm

        if cfg.profile_steps and step == profile_end - 1:
            torch.cuda.profiler.stop()

    torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize(device)  # the run's only sync
    step_ms = [
        s.elapsed_time(e)
        for s, e in zip(start_events, end_events, strict=True)
    ]

    # one device -> host copy of the whole run
    losses_host = losses.tolist()
    grad_norms_host = grad_norms.tolist()

    median_ms = statistics.median(step_ms[cfg.warmup_steps:])
    tokens_per_s = cfg.batch_size * cfg.seq_len / (median_ms / 1e3)
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    print(
        f"loss {losses_host[0]:.4f} -> {losses_host[-1]:.4f} | "
        f"median step {median_ms:.1f} ms | "
        f"{tokens_per_s:,.0f} tokens/s | peak mem {peak_mem_gb:.2f} GB"
    )

    env = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "n_params": n_params,
    }
    metrics = {
        "median_step_ms": median_ms,
        "tokens_per_s": tokens_per_s,
        "peak_mem_gb": peak_mem_gb,
        "step_ms": step_ms,
        "losses": losses_host,
        "grad_norms": grad_norms_host,
    }
    if cfg.out_dir:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        config = {**asdict(cfg), **env}
        (out / "config.json").write_text(json.dumps(config, indent=2))
        (out / "metrics.json").write_text(json.dumps(metrics))

    log_result("train_step", {**asdict(cfg), "n_params": n_params}, **metrics)


def parse_args() -> TrainConfig:
    """Every TrainConfig field becomes a --flag (bools are switches)."""
    parser = argparse.ArgumentParser()
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
