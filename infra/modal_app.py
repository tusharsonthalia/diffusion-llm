"""Modal app: data prep, tests on a GPU, and the training ablation.

Usage (from the repo root):
    uv run modal run infra/modal_app.py::prepare_data --shards 4
    uv run modal run infra/modal_app.py::run_tests
    uv run modal run infra/modal_app.py::ablation --batch-size 8

Every compared benchmark / profile runs on an A100-80GB; tests run on a T4.

Lives in infra/ rather than modal/: a top-level `modal/` directory can
shadow the real `modal` package on sys.path.
"""
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/root/project"

# persistent storage: tokenized data, HF cache, and run outputs
volume = modal.Volume.from_name("diffusion-llm-vol", create_if_missing=True)
VOL = "/vol"
RUNS = f"{VOL}/runs"

# Nsight versions from NVIDIA's Debian 12 CUDA apt repo
NSYS_PKG = "nsight-systems-2026.3.2"
NCU_PKG = "nsight-compute-2026.3.1"
CUDA_KEYRING = (
    "https://developer.download.nvidia.com/compute/cuda/repos/debian12/"
    "x86_64/cuda-keyring_1.1-1_all.deb"
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    # nsys / ncu aren't on PyPI: install from NVIDIA's apt repo, then link
    # the binaries onto PATH (they install under /opt/nvidia/...)
    .apt_install("wget", "ca-certificates")
    .run_commands(
        f"wget -q {CUDA_KEYRING} && dpkg -i cuda-keyring_1.1-1_all.deb"
        " && rm cuda-keyring_1.1-1_all.deb",
        "apt-get update -q && apt-get install -y -q"
        f" --no-install-recommends {NSYS_PKG} {NCU_PKG}"
        " && rm -rf /var/lib/apt/lists/*",
        "ln -sf $(find /opt/nvidia -type f -name nsys -path '*/bin/*'"
        " | head -1) /usr/local/bin/nsys",
        "ln -sf $(find /opt/nvidia -type f -name ncu | head -1)"
        " /usr/local/bin/ncu",
    )
    # Linux torch wheels bundle CUDA and Triton
    .uv_pip_install(
        "torch>=2.14.1", "einops>=0.8.2", "jaxtyping>=0.3.11",
        "pytest>=9.1.1", "numpy",
    )
    .env({
        "HF_HOME": f"{VOL}/hf",
        "PYTHONPATH": REMOTE,
        # Triton's FMA-fusion default flips in 3.10; pin it so kernel
        # rounding doesn't silently change against references
        "TRITON_DEFAULT_FP_FUSION": "1",
    })
    # project code last, so code edits don't invalidate the pip layer
    .add_local_dir(
        REPO,
        remote_path=REMOTE,
        ignore=[".venv", ".git", "ref_repo", "results", "data",
                "**/__pycache__", ".ruff_cache", "*.pdf", ".DS_Store"],
    )
)

# data prep is CPU-only: no torch / Nsight, just the repo's scripts/
data_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "tiktoken>=0.14.0", "pyarrow>=26.0.0", "numpy",
        "huggingface_hub>=2.3.0",
    )
    .env({"HF_HOME": f"{VOL}/hf"})
    .add_local_dir(REPO / "scripts", remote_path=f"{REMOTE}/scripts")
)

app = modal.App("diffusion-llm", image=image)


def _run(cmd: list[str], log_path: str | None = None):
    r = subprocess.run(cmd, cwd=REMOTE, capture_output=True, text=True,
                       check=False)
    if log_path:
        Path(log_path).write_text(r.stdout + r.stderr)
    return r


@app.function(image=data_image, cpu=8, memory=16 * 1024,
              volumes={VOL: volume}, timeout=3 * 60 * 60)
def prepare_data(shards: int = 4, val_docs: int = 5000) -> None:
    """Download + tokenize OpenWebText onto /vol/data/openwebtext (CPU)."""
    subprocess.run(
        ["python", f"{REMOTE}/scripts/prepare_data.py",
         "--out-dir", f"{VOL}/data/openwebtext", "--shards", str(shards),
         "--val-docs", str(val_docs), "--threads", "8"],
        check=True,
    )
    volume.commit()  # visible to other containers right away


@app.function(gpu="T4", volumes={VOL: volume}, timeout=20 * 60)
def run_tests(pytest_args: str = "-q") -> int:
    """Run the test suite on a GPU (Linux, CUDA torch)."""
    cmd = ["python", "-m", "pytest", *pytest_args.split()]
    return subprocess.run(cmd, cwd=REMOTE, check=False).returncode


# ---------------------------------------------------------------------------
# Training ablation: every row on an A100-80GB; each run's config, losses,
# grad norms, step times (and nsys trace) saved under /vol/runs/<pass>/<row>
# ---------------------------------------------------------------------------

# cumulative: each row adds one optimization to the one above
ABLATION = {
    "mdlm_baseline": "--model mdlm",
    "bd3lm_baseline": "--model bd3lm",
    "bd3lm_overlap": "--model bd3lm --step overlap",
    "bd3lm_overlap_fused": "--model bd3lm --step overlap --fused-adam",
    "bd3lm_overlap_fused_sdpa":
        "--model bd3lm --step overlap --fused-adam --attn sdpa",
    "bd3lm_overlap_fused_sdpa_compile":
        "--model bd3lm --step overlap --fused-adam --attn sdpa --compile",
}

# timing pass: 100 steps; nsys pass: a short capture right after warmup,
# with attention annotated (except compiled rows: it would graph-break)
TIMING_ARGS = "--steps 100 --warmup-steps 10"
NSYS_ARGS = "--steps 25 --warmup-steps 10 --profile-steps 10"


def run_args(row: str, nsys: bool, batch_size: int) -> str:
    """Command-line flags for one ABLATION row in one pass."""
    if not nsys:
        return f"{row} {TIMING_ARGS} --batch-size {batch_size}"
    nvtx = "" if "--compile" in row else " --nvtx-attn"
    return f"{row} {NSYS_ARGS}{nvtx} --batch-size {batch_size}"


@app.function(gpu="A100-80GB", volumes={VOL: volume}, timeout=60 * 60)
def train(name: str, args: str, nsys: bool = False) -> dict:
    """One training run; writes /vol/runs/<pass>/<name>/ on the Volume.

    Files: config.json, metrics.json (losses, grad_norms, step_ms, ...),
    log.txt, and for the nsys pass report.nsys-rep + nsys_stats.txt.
    """
    out = f"{RUNS}/{'nsys' if nsys else 'timing'}/{name}"
    Path(out).mkdir(parents=True, exist_ok=True)
    cmd = ["python", "-m", "train.train", *args.split(), "--out-dir", out]

    if nsys:
        cmd = [
            "nsys", "profile", "--trace=cuda,nvtx,osrt",
            "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "--force-overwrite=true", "-o", f"{out}/report", *cmd,
        ]
    r = _run(cmd, f"{out}/log.txt")

    if nsys and r.returncode == 0:
        stats = _run([
            "nsys", "stats", "--force-export=true",
            # nvtx_gpu_proj_sum: GPU time of each NVTX range (attention,
            # forward, ...), measured from the kernels launched inside it
            "--report",
            "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum,"
            "nvtx_gpu_proj_sum",
            f"{out}/report.nsys-rep",
        ])
        Path(f"{out}/nsys_stats.txt").write_text(stats.stdout + stats.stderr)

    volume.commit()
    summary = [ln for ln in r.stdout.splitlines() if ln.startswith("loss ")]
    tail = (r.stdout + r.stderr)[-800:]
    return {"name": name, "exit": r.returncode,
            "summary": summary[0] if summary else tail}


@app.local_entrypoint()
def ablation(batch_size: int, passes: str = "timing,nsys"):
    """Run every ABLATION row (in parallel) for each pass at one batch size."""
    for p in passes.split(","):
        jobs = [(name, run_args(row, p == "nsys", batch_size), p == "nsys")
                for name, row in ABLATION.items()]
        print(f"=== {p} pass ({len(jobs)} runs, batch {batch_size})")
        for res in train.starmap(jobs):
            print(f"{res['name']:34s} exit={res['exit']} | {res['summary']}")
