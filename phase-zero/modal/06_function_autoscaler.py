#!/usr/bin/env python
"""Phase-Zero #6 — @app.function autoscaler: many inputs -> many containers.

Distinct from #5 (imperative Sandbox pool). Here we prove Modal's *managed*
autoscaler: a single decorated function, fanned out with .map() over many inputs,
elastically scales to multiple containers and respects max_containers. This is the
"compute is a decision variable" control surface — the agent tunes min/max/buffer
containers and Modal follows (modal.com/docs/guide/scale).

Verifies:
  - @app.function(max_containers=K) + .map(N inputs) provisions >1 worker task
    (counted by Modal task ids; Modal hostnames are not unique) and never exceeds K
  - .update_autoscaler(...) runtime knob is callable (best-effort demonstration)

Run (after auth):  .venv/bin/modal run phase-zero/modal/06_function_autoscaler.py
Env:  MODAL_VERIFY_MAXC=5   MODAL_VERIFY_NINPUTS=40
"""
from __future__ import annotations
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal
from _common import APP_NAME

MAXC = int(os.environ.get("MODAL_VERIFY_MAXC", "5"))
NINPUTS = int(os.environ.get("MODAL_VERIFY_NINPUTS", "40"))

app = modal.App(f"{APP_NAME}-autoscale")
COMMON_PY = pathlib.Path(__file__).resolve().with_name("_common.py")
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    COMMON_PY, "/root/_common.py", copy=True
)


@app.function(image=image, max_containers=MAXC, scaledown_window=60)
@modal.concurrent(max_inputs=1, target_inputs=1)
def busy(i: int) -> str:
    """A little CPU work so containers stay busy long enough to force scale-out."""
    import json, os, socket, time, hashlib
    from modal import current_function_call_id, current_input_id
    h = hashlib.sha256()
    t = time.time()
    for x in range(4_000_000):
        h.update(str(x).encode())
    time.sleep(float(os.environ.get("MODAL_VERIFY_AUTOSCALE_SLEEP", "3")))
    return json.dumps({
        "i": i,
        "hostname": socket.gethostname(),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "current_input_id": current_input_id(),
        "current_function_call_id": current_function_call_id(),
        "duration_s": round(time.time() - t, 2),
        "hash": h.hexdigest()[:8],
    }, sort_keys=True)


@app.local_entrypoint()
def main() -> None:
    print(f"[06_autoscaler] mapping busy() over {NINPUTS} inputs, max_containers={MAXC} ...", flush=True)
    import json
    results = [json.loads(r) for r in busy.map(range(NINPUTS))]
    task_ids = sorted({r["modal_task_id"] for r in results if r.get("modal_task_id")})
    hostnames = sorted({r["hostname"] for r in results if r.get("hostname")})
    durations = [float(r["duration_s"]) for r in results]
    print(f"[06_autoscaler] {len(results)} results across {len(task_ids)} Modal task ids "
          f"(hostnames observed: {hostnames}):", flush=True)
    for tid in task_ids:
        served = [r["i"] for r in results if r.get("modal_task_id") == tid]
        print(f"    task {tid}: inputs={served}", flush=True)

    assert len(results) == NINPUTS, f"expected {NINPUTS} results, got {len(results)}"
    assert all(r.get("hash") == results[0].get("hash") for r in results), "not all inputs completed the same work"
    assert all(r.get("current_input_id") for r in results), "Modal did not expose current_input_id for every input"
    assert len(task_ids) > 1, "autoscaler did NOT scale out (only 1 Modal task id served all inputs)"
    assert len(task_ids) <= MAXC, f"autoscaler exceeded max_containers={MAXC} (saw {len(task_ids)})"

    # Best-effort runtime knob demonstration (does not gate PASS).
    try:
        busy.update_autoscaler(min_containers=0, max_containers=MAXC, buffer_containers=1)
        print(f"[06_autoscaler] update_autoscaler(min=0,max={MAXC},buffer=1) accepted", flush=True)
    except Exception as e:  # report, do not hide
        print(f"[06_autoscaler] update_autoscaler note (non-fatal in ephemeral run): {e!r}", flush=True)

    print(f"[06_autoscaler] duration range: min={min(durations):.2f}s max={max(durations):.2f}s", flush=True)
    print(f"\033[32mPASS\033[0m  managed autoscaler verified: {NINPUTS} inputs -> "
          f"{len(task_ids)} Modal task ids (1 < {len(task_ids)} <= {MAXC})", flush=True)
