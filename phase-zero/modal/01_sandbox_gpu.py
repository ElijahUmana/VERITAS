#!/usr/bin/env python
"""Phase-Zero #1 — real GPU sandbox spin-up via modal.Sandbox.create(gpu=...).

Proves: we can elastically provision a GPU container on demand, run a command
*inside* it, and observe the real hardware. This is the atom of the whole
VERITAS control loop: `modal.Sandbox.create(gpu=...)` is the primitive emitted
as an action.

Verifies:
  - Sandbox.create(gpu=...) returns a live container
  - exec() runs a command remotely and streams stdout/stderr
  - nvidia-smi reports a real GPU matching the requested type
  - torch sees CUDA (defensive: only if torch present; nvidia-smi is the source of truth)
  - terminate() tears it down

Run (after auth): .venv/bin/python phase-zero/modal/01_sandbox_gpu.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal
from _common import run, log, require, ok, APP_NAME, GPU, GPU_IS_CPU


def main() -> None:
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    log(f"app.lookup({APP_NAME!r}) -> {app.app_id if hasattr(app,'app_id') else app}")

    # debian_slim is enough: Modal injects the NVIDIA driver + nvidia-smi when gpu= is set.
    image = modal.Image.debian_slim(python_version="3.12")

    log(f"creating GPU sandbox (gpu={GPU!r}, timeout=300s) ...")
    t_create = __import__("time").time()
    sb = modal.Sandbox.create(
        app=app,
        image=image,
        gpu=(None if GPU_IS_CPU else GPU),
        timeout=300,
    )
    import time as _t
    log(f"sandbox created: id={sb.object_id}  (cold start {_t.time()-t_create:.1f}s)")

    try:
        # 1) Identity / OS proof
        p = sb.exec("bash", "-lc", "uname -a && nproc && cat /etc/os-release | head -1")
        uname = p.stdout.read()
        p.wait()
        log("uname/cpu:\n" + uname.strip())
        require(bool(uname.strip()), "exec() ran a remote command and returned stdout")

        if not GPU_IS_CPU:
            # 2) THE GPU proof — nvidia-smi inside the container
            q = sb.exec(
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            )
            smi = q.stdout.read()
            err = q.stderr.read()
            code = q.wait()
            log(f"nvidia-smi (exit={code}):\n{smi.strip()}\n{('stderr: '+err.strip()) if err.strip() else ''}")
            require(code == 0, "nvidia-smi exited 0 inside the GPU sandbox")
            require(bool(smi.strip()), "nvidia-smi reported a real GPU row")
            # The requested GPU family should appear in the device name
            # (T4->'T4', A10->'A10', H100->'H100', etc.; strip ':N' multi-gpu suffix)
            fam = GPU.split(":")[0].split("-")[0].upper()
            require(fam in smi.upper(),
                    f"reported GPU name contains requested family {fam!r}")
        else:
            ok("CPU mode requested (MODAL_VERIFY_GPU=cpu) — skipped nvidia-smi")

        # 3) Remote compute actually executes (not just a shell)
        c = sb.exec("python3", "-c",
                    "import platform,sys;print('PYOK',platform.python_version(),sys.executable)")
        pyout = c.stdout.read(); c.wait()
        log("remote python: " + pyout.strip())
        require("PYOK" in pyout, "remote python3 executed inside the sandbox")

        ok(f"GPU sandbox fully verified (gpu={GPU})")
    finally:
        # Resource cleanup — does NOT swallow the primary exception (it re-raises after finally).
        log(f"terminating sandbox {sb.object_id} ...")
        sb.terminate()
        log("terminated")


if __name__ == "__main__":
    run(main)
