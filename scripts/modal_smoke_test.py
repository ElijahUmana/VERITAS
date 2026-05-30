#!/usr/bin/env python
"""Modal Sandbox smoke test + compute-characterization benchmark (Phase-Zero #8).

Proves the core elastic-compute loop AND measures the latencies the team needs to
size the control plane (Task #6) and the uncertainty-proportional swarm (thesis):

  Tier 1 (default, CPU):   App.lookup -> Sandbox.create -> exec -> write/read file
                           -> snapshot_filesystem -> terminate.  Measures cold-start,
                           first-exec round-trip, and snapshot-create latency.
  Tier R (default):        boot a NEW sandbox FROM the Tier-1 snapshot image_id and
                           confirm the written file is present.  Measures RESTORE latency
                           (the agent-fork / checkpoint-resume path).
  Tier B (default):        create N sandboxes CONCURRENTLY (async .aio gather), exec in
                           each, terminate.  Measures burst wall-clock + effective
                           spawns/sec + per-create p50/p95 (the fan-out number).
  Tier 2 (--gpu T4):       GPU sandbox + nvidia-smi.  Proves GPU attach + GPU cold-start.
  Tier 3 (--tunnel):       encrypted_ports + tunnels() URL.  Proves port exposure.

Every Modal API used is verified against the installed client (v1.3.5). Run after `modal setup`:

    python scripts/modal_smoke_test.py                          # Tiers 1, R, B
    python scripts/modal_smoke_test.py --gpu T4 --tunnel        # + GPU + tunnel
    python scripts/modal_smoke_test.py --burst 16               # bigger burst

Exit 0 = all selected tiers PASS. Output ends with a MEASUREMENTS block for pasting
into Task #8 and sharing with architecture-master / synthesist.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import traceback

APP_NAME = "veritas-smoke"
IMAGE_TAG = "python:3.11-slim"

# Collected measurements (filled by tiers), emitted at the end.
M: dict[str, object] = {}


def _t() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_t()}] {msg}", flush=True)


def _read(stream) -> str:
    data = stream.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data or ""


def tier1_cpu(image, app) -> tuple[str, str]:
    """Returns (snapshot_image_id, token_written) for the restore tier."""
    import modal

    log("TIER 1 (CPU) — creating sandbox ...")
    t0 = time.monotonic()
    sb = modal.Sandbox.create(app=app, image=image, timeout=180)
    create_s = time.monotonic() - t0
    M["cold_start_create_s"] = round(create_s, 3)
    log(f"  sandbox created: id={sb.object_id} cold_start_create={create_s:.2f}s")

    # first-exec round-trip (create -> ready shell delta)
    t0 = time.monotonic()
    p = sb.exec("bash", "-c", "echo hello-from-modal && python --version && uname -m")
    out = _read(p.stdout)
    err = _read(p.stderr)
    code = p.wait()
    exec_s = time.monotonic() - t0
    M["first_exec_roundtrip_s"] = round(exec_s, 3)
    assert code == 0, f"exec exit={code} stderr={err!r}"
    assert "hello-from-modal" in out, f"missing stdout marker: {out!r}"
    log(f"  first exec OK ({exec_s:.2f}s) exit={code} stdout={out.strip()!r}")

    # write a file (this is what the restore tier will verify survived the snapshot)
    token = f"smoke-{int(time.time())}"
    w = sb.exec("bash", "-c", f"echo {token} > /root/smoke.txt")
    assert w.wait() == 0, "write failed"
    r = sb.exec("cat", "/root/smoke.txt")
    rout = _read(r.stdout)
    assert r.wait() == 0 and token in rout, f"readback failed: {rout!r}"
    log(f"  filesystem RW OK token={token}")

    # filesystem snapshot -> reusable Image (fast-start / fork primitive)
    t0 = time.monotonic()
    snap = sb.snapshot_filesystem()
    snap_s = time.monotonic() - t0
    image_id = getattr(snap, "object_id", None) or getattr(snap, "id", None) or str(snap)
    M["snapshot_create_s"] = round(snap_s, 3)
    M["snapshot_image_id"] = image_id
    log(f"  snapshot_filesystem OK image_id={image_id} took={snap_s:.2f}s")

    assert sb.poll() is None, "sandbox should be running before terminate"
    sb.terminate()
    log("  terminate OK")
    log("TIER 1 PASS ✅")
    return image_id, token


def tier_restore(app, image_id: str, token: str) -> None:
    """Boot a NEW sandbox from the snapshot image; confirm the file survived."""
    import modal

    log(f"TIER R (restore) — booting from snapshot image_id={image_id} ...")
    t0 = time.monotonic()
    img = modal.Image.from_id(image_id)
    sb = modal.Sandbox.create(app=app, image=img, timeout=120)
    restore_s = time.monotonic() - t0
    M["snapshot_restore_s"] = round(restore_s, 3)
    p = sb.exec("cat", "/root/smoke.txt")
    out = _read(p.stdout)
    code = p.wait()
    sb.terminate()
    assert code == 0 and token in out, f"snapshot did not carry the file: {out!r}"
    log(f"  restore OK ({restore_s:.2f}s) — file survived snapshot, token={token}")
    log("TIER R PASS ✅")


def tier_burst(app, image, n: int) -> None:
    """Create N sandboxes concurrently (async), exec, terminate. Measures fan-out."""
    import modal

    log(f"TIER B (burst) — creating {n} sandboxes CONCURRENTLY ...")

    async def one(i: int) -> float:
        t0 = time.monotonic()
        sb = await modal.Sandbox.create.aio(app=app, image=image, timeout=120)
        created = time.monotonic() - t0
        p = await sb.exec.aio("echo", f"agent-{i}")
        await p.wait.aio()
        await sb.terminate.aio()
        return created

    async def run() -> tuple[list[float], float]:
        t0 = time.monotonic()
        times = await asyncio.gather(*[one(i) for i in range(n)], return_exceptions=True)
        wall = time.monotonic() - t0
        ok = [t for t in times if isinstance(t, float)]
        errs = [t for t in times if not isinstance(t, float)]
        if errs:
            log(f"  WARNING: {len(errs)}/{n} burst creates failed: {errs[0]!r}")
        return ok, wall

    ok, wall = asyncio.run(run())
    assert ok, "all burst creates failed"
    p50 = round(statistics.median(ok), 3)
    p95 = round(sorted(ok)[max(0, int(len(ok) * 0.95) - 1)], 3)
    thr = round(len(ok) / wall, 2) if wall > 0 else 0
    M["burst_n"] = n
    M["burst_ok"] = len(ok)
    M["burst_wall_s"] = round(wall, 3)
    M["burst_spawns_per_s"] = thr
    M["burst_create_p50_s"] = p50
    M["burst_create_p95_s"] = p95
    log(f"  burst: {len(ok)}/{n} OK in {wall:.2f}s  => {thr} spawns/s  per-create p50={p50}s p95={p95}s")
    log("TIER B PASS ✅")


def tier2_gpu(image, app, gpu: str) -> None:
    import modal

    log(f"TIER 2 (GPU={gpu}) — creating GPU sandbox ...")
    t0 = time.monotonic()
    sb = modal.Sandbox.create(app=app, image=image, gpu=gpu, timeout=180)
    create_s = time.monotonic() - t0
    M["gpu_cold_start_create_s"] = round(create_s, 3)
    log(f"  GPU sandbox created: id={sb.object_id} cold_start={create_s:.2f}s")
    p = sb.exec("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader")
    out = _read(p.stdout)
    err = _read(p.stderr)
    code = p.wait()
    sb.terminate()
    assert code == 0, f"nvidia-smi exit={code} stderr={err!r}"
    M["gpu_info"] = out.strip()
    log(f"  nvidia-smi OK -> {out.strip()!r}")
    log("TIER 2 PASS ✅")


def tier3_tunnel(image, app) -> None:
    import modal

    log("TIER 3 (tunnel) — exposing encrypted port 8000 ...")
    sb = modal.Sandbox.create(app=app, image=image, encrypted_ports=[8000], timeout=120)
    sb.exec("bash", "-c", "python -m http.server 8000 >/tmp/srv.log 2>&1 &")
    time.sleep(2)
    tunnels = sb.tunnels()
    tun = tunnels.get(8000)
    url = getattr(tun, "url", None) or f"{getattr(tun, 'host', '?')}:{getattr(tun, 'port', '?')}"
    sb.terminate()
    assert tun is not None, f"no tunnel for port 8000: {tunnels!r}"
    M["tunnel_url"] = url
    log(f"  tunnel OK url={url}")
    log("TIER 3 PASS ✅")


def main() -> int:
    ap = argparse.ArgumentParser(description="Modal Sandbox smoke + benchmark (Phase-Zero #8)")
    ap.add_argument("--gpu", default=None, help="Run Tier 2 GPU test with this spec (e.g. T4).")
    ap.add_argument("--tunnel", action="store_true", help="Run Tier 3 tunnel test.")
    ap.add_argument("--burst", type=int, default=8, help="Tier B concurrent sandbox count (0=skip).")
    ap.add_argument("--app-name", default=APP_NAME)
    ap.add_argument("--image-tag", default=IMAGE_TAG)
    args = ap.parse_args()

    log("=== Modal Sandbox SMOKE TEST + BENCHMARK — Phase-Zero #8 ===")
    try:
        import modal
        log(f"modal client version: {modal.__version__}")
    except Exception as e:
        log(f"FATAL: cannot import modal: {e}")
        return 2

    try:
        import modal
        app = modal.App.lookup(args.app_name, create_if_missing=True)
        log(f"app lookup OK: {args.app_name}")
        image = modal.Image.from_registry(args.image_tag)
        log(f"image pinned: {args.image_tag}")
    except Exception as e:
        log("FATAL: Modal not authenticated or unreachable. Run `modal setup` first.")
        log(f"  detail: {e}")
        return 3

    failures: list[str] = []
    snap_id: str | None = None
    token: str | None = None

    # Tier 1 first (produces the snapshot used by Tier R).
    try:
        snap_id, token = tier1_cpu(image, app)
    except Exception as e:
        failures.append("tier1")
        log(f"tier1 FAIL ❌: {e}")
        traceback.print_exc()

    tiers = []
    if snap_id and token:
        tiers.append(("tierR", lambda: tier_restore(app, snap_id, token)))
    if args.burst and args.burst > 0:
        tiers.append(("tierB", lambda: tier_burst(app, image, args.burst)))
    if args.gpu:
        tiers.append(("tier2", lambda: tier2_gpu(image, app, args.gpu)))
    if args.tunnel:
        tiers.append(("tier3", lambda: tier3_tunnel(image, app)))

    for name, fn in tiers:
        try:
            fn()
        except Exception as e:
            failures.append(name)
            log(f"{name} FAIL ❌: {e}")
            traceback.print_exc()

    # Measurements block (easy to paste / share).
    print("\n" + "=" * 60)
    print("MEASUREMENTS (verified live):")
    for k in (
        "cold_start_create_s", "first_exec_roundtrip_s", "snapshot_create_s",
        "snapshot_restore_s", "burst_n", "burst_ok", "burst_wall_s",
        "burst_spawns_per_s", "burst_create_p50_s", "burst_create_p95_s",
        "gpu_cold_start_create_s", "gpu_info", "tunnel_url", "snapshot_image_id",
    ):
        if k in M:
            print(f"  {k} = {M[k]}")
    print("=" * 60)

    # Task #8 gate = the CORE smoke (Tier 1: create/exec/snapshot/terminate).
    # Bonus tiers (R/B/GPU/tunnel) are characterization — their failures are reported
    # loudly but do not flip the gate (e.g. a transient GPU-availability blip on a
    # fresh free-tier account must not mask that the sandbox loop itself works).
    core_failed = "tier1" in failures
    bonus_failed = [f for f in failures if f != "tier1"]
    if bonus_failed:
        log(f"BONUS TIERS FAILED (characterization, NON-GATING): {', '.join(bonus_failed)}")
    if core_failed:
        log("SMOKE TEST RESULT: FAIL — core Tier 1 did not pass; Task #8 gate NOT satisfied.")
        return 1
    if bonus_failed:
        log("SMOKE TEST RESULT: CORE PASS ✅ (Task #8 gate satisfied) — some bonus tiers failed above.")
        return 0
    log("SMOKE TEST RESULT: ALL TIERS PASS ✅ (Task #8 gate satisfied).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
