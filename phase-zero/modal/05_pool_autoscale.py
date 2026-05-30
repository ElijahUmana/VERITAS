#!/usr/bin/env python
"""Phase-Zero #5 — elastic pool fan-out.

Provision a pool of sandboxes concurrently, run work in each, and measure
parallel overlap. The timing comparison below uses concurrently observed
per-sandbox elapsed times; it is not an independently measured serial baseline.
This is the population substrate the VERITAS control loop scales.

Verifies:
  - N sandboxes provision concurrently (Sandbox.create is the unit of horizontal scale)
  - each sandbox has a unique Modal sandbox ID
  - each runs real work and returns a result
  - real GPU mode runs a fast nvidia-smi query inside every sandbox
  - measured parallel wall-clock << sum of concurrently measured per-sandbox times
    (overlap evidence, not a separately measured serial control)
  - reports observed pool-concurrency headroom (GPU quota headroom in GPU mode)

Run (after auth):  .venv/bin/python phase-zero/modal/05_pool_autoscale.py
Env:  MODAL_VERIFY_POOL=8   (pool size; keep modest on a fresh acct's GPU quota)
      MODAL_VERIFY_GPU=cpu  (CPU pool is fine to prove the scaling mechanism cheaply)
"""
from __future__ import annotations
import json, os, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal
from _common import run, log, require, ok, APP_NAME, GPU, GPU_IS_CPU

POOL = int(os.environ.get("MODAL_VERIFY_POOL", "8"))
# A small unit of real work done inside each sandbox (CPU-bound python).
WORK_PY = r"""
import hashlib
import json
import os
import socket
import subprocess
import time

t = time.time()
h = hashlib.sha256()
for i in range(1_500_000):
    h.update(str(i).encode())

info = {
    "worked": True,
    "hash": h.hexdigest()[:12],
    "hostname": socket.gethostname(),
    "work_s": round(time.time() - t, 2),
    "nvidia_smi": None,
    "nvidia_smi_err": "",
    "nvidia_smi_rc": None,
}

if os.environ.get("VERIFY_NVIDIA_SMI") == "1":
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info["nvidia_smi_rc"] = p.returncode
        info["nvidia_smi"] = (p.stdout.strip().splitlines() or [""])[0]
        info["nvidia_smi_err"] = p.stderr.strip()
    except Exception as exc:
        info["nvidia_smi_rc"] = -1
        info["nvidia_smi_err"] = repr(exc)

print(json.dumps(info, sort_keys=True))
"""


def work_command(verify_gpu: bool) -> str:
    verify = "1" if verify_gpu else "0"
    return f"VERIFY_NVIDIA_SMI={verify} python3 - <<'PY'\n{WORK_PY}\nPY"


def main() -> None:
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12")
    gpu = None if GPU_IS_CPU else GPU
    verify_gpu = not GPU_IS_CPU
    work = work_command(verify_gpu)

    require(POOL >= 2, "pool size is at least 2 so overlap can be measured")

    def one(i: int) -> dict:
        t0 = time.time()
        sb = modal.Sandbox.create(app=app, image=image, gpu=gpu, timeout=300)
        boot = time.time() - t0
        try:
            p = sb.exec("bash", "-lc", work)
            out = p.stdout.read().strip()
            code = p.wait()
            try:
                payload = json.loads(out.splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                payload = {"parse_error": repr(exc)}
            return {"i": i, "id": sb.object_id, "boot_s": boot,
                    "total_s": time.time() - t0, "out": out,
                    "payload": payload, "code": code}
        finally:
            sb.terminate()

    # --- Parallel fan-out: provision the whole pool at once --------------------
    smi_mode = "on" if verify_gpu else "skipped"
    log(f"fanning out a pool of {POOL} sandboxes in parallel "
        f"(gpu={GPU}; per-sandbox nvidia-smi={smi_mode}) ...")
    from concurrent.futures import ThreadPoolExecutor
    t_par = time.time()
    with ThreadPoolExecutor(max_workers=POOL) as ex:
        results = list(ex.map(one, range(POOL)))
    par_wall = time.time() - t_par

    for r in results:
        payload = r["payload"]
        details = (f"hash={payload.get('hash', '?')} "
                   f"work={payload.get('work_s', '?')}s")
        if verify_gpu:
            details += (f" nvidia_smi_rc={payload.get('nvidia_smi_rc')} "
                        f"nvidia_smi={payload.get('nvidia_smi')!r}")
        if "parse_error" in payload:
            details += f" parse_error={payload['parse_error']} raw={r['out']!r}"
        log(f"  sb {r['i']}: id={r['id']} boot={r['boot_s']:.1f}s "
            f"total={r['total_s']:.1f}s code={r['code']} :: {details}")

    ids = [r["id"] for r in results]
    require(all(ids), f"all {POOL} pooled sandboxes returned Modal sandbox IDs")
    require(len(set(ids)) == POOL,
            f"all {POOL} pooled sandboxes had unique Modal sandbox IDs")
    require(all(r["code"] == 0 for r in results), f"all {POOL} pooled sandboxes exited 0")
    require(all(r["payload"].get("worked") is True for r in results),
            f"all {POOL} sandboxes did real work")
    if verify_gpu:
        require(all(r["payload"].get("nvidia_smi_rc") == 0 for r in results),
                f"nvidia-smi exited 0 inside all {POOL} GPU sandboxes")
        require(all(str(r["payload"].get("nvidia_smi") or "").strip() for r in results),
                f"nvidia-smi reported a GPU row inside all {POOL} GPU sandboxes")

    sum_member_elapsed = sum(r["total_s"] for r in results)
    max_member_elapsed = max(r["total_s"] for r in results)
    overlap = sum_member_elapsed / par_wall if par_wall else 0.0
    wall_vs_slowest = par_wall / max_member_elapsed if max_member_elapsed else 0.0
    log(f"parallel wall-clock={par_wall:.1f}s   "
        f"sum-of-concurrent-member-elapsed={sum_member_elapsed:.1f}s   "
        f"slowest-member={max_member_elapsed:.1f}s   "
        f"overlap_factor={overlap:.1f}x   wall/slowest={wall_vs_slowest:.1f}x")
    log("note: overlap_factor is computed from concurrent member timings; "
        "this run does not measure a separate serial baseline.")
    require(overlap > 1.5,
            f"parallel overlap observed (overlap_factor {overlap:.1f}x > 1.5x "
            "using concurrent member timings)")
    ok(f"ELASTIC POOL VERIFIED: {POOL} unique concurrent sandboxes, "
       f"{overlap:.1f}x measured overlap")
    log("note: this is the blog's '~40 parallel sandboxes' mechanism at N=%d. "
        "Scale N with GPU quota (fresh Starter=10 GPU concurrency, Team=50)." % POOL)


if __name__ == "__main__":
    run(main)
