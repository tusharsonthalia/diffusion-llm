"""Benchmark + correctness harness: bench, check, log_result."""
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


@dataclass
class Timing:
    median_ms: float
    p20_ms: float
    p80_ms: float
    n_iters: int


def bench(fn: Callable[[], Any], warmup: int = 25, rep: int = 100) -> Timing:
    """Time fn() over `rep` calls after `warmup` untimed ones (JIT, autotuning, allocator)."""
    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # zeroing a buffer larger than L2 evicts it, so each rep reads inputs from HBM
        l2 = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(rep)]
        for start, end in events:
            l2.zero_()
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        times = [start.elapsed_time(end) for start, end in events]
    else:
        times = []
        for _ in range(rep):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)

    p20, p50, p80 = torch.tensor(times).quantile(torch.tensor([0.2, 0.5, 0.8])).tolist()
    return Timing(median_ms=p50, p20_ms=p20, p80_ms=p80, n_iters=rep)


@dataclass
class Check:
    passed: bool
    max_abs: float
    max_rel: float
    cosine: float


def check(out: Tensor, ref: Tensor, atol: float = 1e-5, rtol: float = 1e-4) -> Check:
    """Compare in float64; pass iff every |out - ref| <= atol + rtol * |ref| (a NaN fails)."""
    o, r = out.detach().double().flatten(), ref.detach().double().flatten().to(out.device)
    diff = (o - r).abs()
    return Check(
        passed=bool((diff <= atol + rtol * r.abs()).all()),
        max_abs=diff.max().item(),
        # informational only: blows up where ref ~ 0, which is why pass/fail uses atol + rtol
        max_rel=(diff / r.abs().clamp_min(1e-12)).max().item(),
        cosine=torch.nn.functional.cosine_similarity(o, r, dim=0).item(),
    )


def log_result(name: str, config: dict, timing: Timing | None = None,
               check: Check | None = None, **metrics: Any) -> None:
    """Append one run (config, timing, check, environment) to RESULTS_DIR/<name>.jsonl."""
    record = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
        "timing": asdict(timing) if timing else None,
        "check": asdict(check) if check else None,
        "env": {
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        **metrics,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / f"{name}.jsonl").open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
