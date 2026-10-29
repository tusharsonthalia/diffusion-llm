"""Modal app for GPU runs: environment checks, tests, and (later) benchmarks/profiles.

Usage (from the repo root):
    uv run modal run infra/modal_app.py                 # env check + tests on a T4
    uv run modal run infra/modal_app.py::env_check      # just the GPU / Triton / profiler check
    uv run modal run infra/modal_app.py::run_tests      # just the test suite
    uv run modal run infra/modal_app.py::nsight_check   # nsys + ncu on a real kernel
    uv run modal run infra/modal_app.py::prepare_data --shards 4   # OpenWebText -> /vol/data (CPU)

GPU policy (PLAN.md): cheap GPUs (T4 / L4 / A10G) for correctness, H100 only for final
benchmarks and profiles. Override per call with GPU=L4 uv run modal run ....

Lives in infra/ rather than modal/: a top-level `modal/` directory can shadow the real
`modal` package on sys.path.
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/root/project"
GPU = os.environ.get("GPU", "T4")

# persistent storage: HF cache (tokenizers, GPT-2 Large, checkpoints) and results
volume = modal.Volume.from_name("diffusion-llm-vol", create_if_missing=True)
VOL = "/vol"

# Nsight versions from NVIDIA's Debian 12 CUDA apt repo (newest available at setup time)
NSYS_PKG = "nsight-systems-2026.3.2"
NCU_PKG = "nsight-compute-2026.3.1"

image = (
    modal.Image.debian_slim(python_version="3.13")
    # Nsight Systems (nsys) + Nsight Compute (ncu): not on PyPI, so install from NVIDIA's
    # apt repo, then link the binaries onto PATH (they install under /opt/nvidia/...)
    .apt_install("wget", "ca-certificates")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
        " && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb",
        f"apt-get update -q && apt-get install -y -q --no-install-recommends {NSYS_PKG} {NCU_PKG}"
        " && rm -rf /var/lib/apt/lists/*",
        "ln -sf $(find /opt/nvidia -type f -name nsys -path '*/bin/*' | head -1) /usr/local/bin/nsys",
        "ln -sf $(find /opt/nvidia -type f -name ncu | head -1) /usr/local/bin/ncu",
        "nsys --version && ncu --version",
    )
    # Linux torch wheels bundle CUDA and Triton
    .uv_pip_install("torch>=2.14.1", "einops>=0.8.2", "jaxtyping>=0.3.11", "pytest>=9.1.1", "numpy")
    .env({
        "HF_HOME": f"{VOL}/hf",
        "PYTHONPATH": REMOTE,
        # pin Triton's FMA fusion explicitly: its default flips in Triton 3.10, which would
        # silently change kernel rounding vs. references
        "TRITON_DEFAULT_FP_FUSION": "1",
    })
    # project code last, so code edits don't invalidate the cached pip layer
    .add_local_dir(
        REPO,
        remote_path=REMOTE,
        ignore=[".venv", ".git", "ref_repo", "results", "**/__pycache__", ".ruff_cache", "*.pdf", ".DS_Store"],
    )
)

app = modal.App("diffusion-llm", image=image)


@app.function(gpu=GPU, volumes={VOL: volume}, timeout=10 * 60)
def env_check() -> dict:
    """Confirm the GPU, CUDA torch, Triton (a real kernel), and torch.profiler all work."""
    import shutil
    import subprocess

    import torch
    import triton
    import triton.language as tl

    info = {
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "nsys_available": shutil.which("nsys") is not None,
        "ncu_available": shutil.which("ncu") is not None,
    }
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout)

    # a minimal Triton kernel, checked against torch
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

    n = 98_432
    x, y = torch.rand(n, device="cuda"), torch.rand(n, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    info["triton_kernel_ok"] = bool(torch.allclose(out, x + y))

    # torch.profiler with CUDA activity (the fallback if Nsight counters are unavailable)
    a = torch.randn(1024, 1024, device="cuda")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(3):
            a @ a
        torch.cuda.synchronize()
    info["profiler_cuda_events"] = sum(1 for e in prof.events() if e.device_type.name == "CUDA")

    for k, v in info.items():
        print(f"{k:>22}: {v}")
    return info


# a tiny profiling target: one Triton kernel + one matmul, with an NVTX range
_PROFILE_TARGET = """
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

x = torch.rand(1 << 20, device="cuda"); y = torch.rand_like(x); out = torch.empty_like(x)
with torch.cuda.nvtx.range("profile_target"):
    add_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK=1024)
    a = torch.randn(1024, 1024, device="cuda"); a @ a
torch.cuda.synchronize()
"""


@app.function(gpu=GPU, volumes={VOL: volume}, timeout=15 * 60)
def nsight_check() -> dict:
    """Profile a real Triton kernel with nsys (timeline) and ncu (hardware counters).

    nsys only needs CUDA tracing. ncu reads GPU performance counters, which a container
    may not be allowed to access (ERR_NVGPUCTRPERM); if so, fall back to torch.profiler
    + do_bench for kernel metrics (PLAN.md 0.4).
    """
    import subprocess
    from pathlib import Path

    work = Path("/tmp/nsight_check")
    work.mkdir(exist_ok=True)
    (work / "target.py").write_text(_PROFILE_TARGET)

    def run(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=work, capture_output=True, text=True, check=False)

    nsys = run(["nsys", "profile", "--trace=cuda,nvtx", "--force-overwrite=true", "-o", "report",
                "python", "target.py"])
    nsys_stats = run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "report.nsys-rep"])
    # --clock-control none: a container can't lock GPU clocks (ncu's default), so profile
    # at the GPU's current clocks -- fine for relative comparisons, note it in results
    ncu = run(["ncu", "--clock-control", "none", "--kernel-name", "regex:add_kernel",
               "--section", "SpeedOfLight", "python", "target.py"])

    result = {
        "nsys_ok": nsys.returncode == 0 and (work / "report.nsys-rep").exists(),
        # ncu prints the section as "GPU Speed Of Light Throughput"
        "ncu_ok": ncu.returncode == 0 and "Speed Of Light" in ncu.stdout
                  and "ERR_NVGPUCTRPERM" not in ncu.stdout + ncu.stderr,
        "ncu_counters_blocked": "ERR_NVGPUCTRPERM" in ncu.stdout + ncu.stderr,
    }
    print("=== nsys stats (GPU kernel summary) ===\n", nsys_stats.stdout[-2500:] or nsys.stderr[-2500:])
    print("=== ncu ===\n", (ncu.stdout + ncu.stderr)[-2500:])
    for k, v in result.items():
        print(f"{k:>22}: {v}")
    return result


@app.function(gpu=GPU, volumes={VOL: volume}, timeout=20 * 60)
def run_tests(pytest_args: str = "-q") -> int:
    """Run the project's test suite inside the GPU container (Linux, CUDA torch)."""
    import subprocess

    result = subprocess.run(["python", "-m", "pytest", *pytest_args.split()], cwd=REMOTE, check=False)
    return result.returncode


# data prep is CPU-only: a slim image without torch / Nsight, plus the repo's scripts/
data_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("tiktoken>=0.14.0", "pyarrow>=26.0.0", "numpy", "huggingface_hub>=2.3.0")
    .env({"HF_HOME": f"{VOL}/hf"})
    .add_local_dir(REPO / "scripts", remote_path=f"{REMOTE}/scripts")
)


@app.function(image=data_image, cpu=8, memory=16 * 1024, volumes={VOL: volume}, timeout=3 * 60 * 60)
def prepare_data(shards: int = 4, val_docs: int = 5000) -> None:
    """Download + tokenize OpenWebText onto the Volume at /vol/data/openwebtext (CPU only)."""
    import subprocess

    subprocess.run(
        ["python", f"{REMOTE}/scripts/prepare_data.py", "--out-dir", f"{VOL}/data/openwebtext",
         "--shards", str(shards), "--val-docs", str(val_docs), "--threads", "8"],
        check=True,
    )
    volume.commit()  # make the files visible to other containers right away


@app.local_entrypoint()
def main():
    info = env_check.remote()
    assert info["triton_kernel_ok"], "Triton kernel produced wrong results"
    nsight_check.remote()
    code = run_tests.remote()
    print(f"\npytest exit code: {code}")
