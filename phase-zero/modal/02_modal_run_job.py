#!/usr/bin/env python
"""Phase-Zero #2 — REAL `modal run` job on a GPU function (@app.function).

Proves the *other* Modal idiom (used by the modal-gpu-experiment skill): a
decorated remote function with a GPU, invoked from a local entrypoint, with
retries configured. This is the batch-experiment path the swarm uses for
training/eval jobs (vs. the interactive Sandbox path in #1).

Verifies:
  - @app.function(gpu=..., retries=...) provisions a GPU worker
  - .remote() executes it and returns a real value to the client
  - the returned payload reports the real GPU + a tiny CUDA tensor op if torch exists
  - `modal run` ephemeral-app lifecycle works end to end

Run (after auth):  .venv/bin/modal run phase-zero/modal/02_modal_run_job.py
(equivalently)     .venv/bin/python -m modal run phase-zero/modal/02_modal_run_job.py
"""
from __future__ import annotations
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal
from _common import APP_NAME, GPU, GPU_IS_CPU

app = modal.App(f"{APP_NAME}-run")
COMMON_PY = pathlib.Path(__file__).resolve().with_name("_common.py")

# Keep image light. torch is optional — we add it only when a GPU is requested so the
# job can prove a real CUDA tensor op, but nvidia-smi via subprocess is the source of truth.
_image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    COMMON_PY, "/root/_common.py", copy=True
)
if not GPU_IS_CPU:
    _image = _image.pip_install("torch", "numpy")


@app.function(
    image=_image,
    gpu=(None if GPU_IS_CPU else GPU),
    timeout=600,
    retries=modal.Retries(max_retries=2),  # fault tolerance, per modal-gpu-experiment skill
)
def gpu_probe() -> dict:
    """Runs INSIDE the Modal GPU worker; returns real hardware facts to the client."""
    import subprocess, platform, json
    info: dict = {"python": platform.python_version(), "node": platform.node()}
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        info["nvidia_smi_rc"] = smi.returncode
        info["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
    except FileNotFoundError:
        info["nvidia_smi"] = "<nvidia-smi not found — CPU container>"
        info["nvidia_smi_rc"] = -1
    # Real CUDA tensor op (only meaningful with a GPU + torch)
    try:
        import torch
        info["torch"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = str(torch.cuda.get_device_name(0))
            x = torch.randn(2048, 2048, device="cuda")
            y = (x @ x).sum().item()  # forces a real GPU matmul
            info["matmul_sum_finite"] = bool(y == y)  # NaN check
    except Exception as e:  # report, never hide
        info["torch_error"] = repr(e)
    return json.loads(json.dumps(info, default=str))


@app.local_entrypoint()
def main() -> None:
    print(f"[02_modal_run_job] invoking gpu_probe.remote() on gpu={GPU!r} ...", flush=True)
    info = gpu_probe.remote()
    import json
    print("[02_modal_run_job] REMOTE RESULT:\n" + json.dumps(info, indent=2), flush=True)

    # Assertions — fail loud
    assert info.get("python"), "remote function returned no python version"
    if not GPU_IS_CPU:
        assert info.get("nvidia_smi_rc") == 0, f"nvidia-smi failed in worker: {info.get('nvidia_smi')}"
        fam = GPU.split(":")[0].split("-")[0].upper()
        assert fam in (info.get("nvidia_smi", "")).upper(), \
            f"worker GPU {info.get('nvidia_smi')!r} does not match requested {fam!r}"
        # If torch installed, CUDA must be live and the matmul finite
        if "cuda_available" in info:
            assert info["cuda_available"], "torch present but CUDA not available in GPU worker"
            assert info.get("matmul_sum_finite"), "GPU matmul produced NaN/!finite"
    print(f"\033[32mPASS\033[0m  modal run GPU job verified live (gpu={GPU})", flush=True)
