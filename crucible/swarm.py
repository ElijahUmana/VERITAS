#!/usr/bin/env python3
"""crucible/swarm.py — CEILING: verified swarm fan-out (Task #11).

Fan out N gpt-5.4-mini-generated RMSNorm candidates IN PARALLEL, each routed
through the SAME real CRUCIBLE gate (Orchestrator + KernelOracle on Modal). Only
SURVIVORS — oracle-confirmed, no tamper, speedup ≥ threshold, detector-D trace
readback — commit to the shared verified ledger. Every candidate lands in the
Raindrop swarm courtroom under one shared ``crucible.mission_id``.

This is the "1000-agent megastructure runs genuine hypothesis → verify →
compound campaigns" beat at demo scale: real breadth, a real verified-survivor
rate, real cost — NO canned verdicts, no trust shortcut.

Parallelism:
  * PROPOSE — N concurrent gpt-5.4-mini structured-output calls (asyncio),
    each with a DISTINCT approach directive so the survivor rate is meaningful.
  * VERIFY  — each candidate routed through its own Orchestrator (sharing the
    swarm ``mission_id``) on a worker thread; the blocking Modal ``.remote``
    calls fan out across concurrent T4 containers (deploy-once-call-many).
    Survivors commit to the canonical WAL ledger (``Ledger()``), so they feed
    crucible-core's compounding curve (#9).

Usage:
    .venv/bin/python crucible/swarm.py --n 10
    .venv/bin/python crucible/swarm.py --n 8 --ledger /tmp/swarm.db --reset-ledger
    .venv/bin/python crucible/swarm.py --n 12 --json
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crucible.generator import DEFAULT_MODEL, propose_candidate  # noqa: E402
from crucible.schemas import Claim, new_id  # noqa: E402

# Modal T4 ≈ $0.59/hr ≈ $0.000164/s (override via env). gpt-5.4-mini per-candidate
# proposal ≈ a cent (rough). Both clearly-labeled ESTIMATES in the report.
T4_USD_PER_SEC = float(os.environ.get("VERITAS_T4_USD_PER_SEC", "0.000164"))
GEN_USD_PER_CANDIDATE = float(os.environ.get("VERITAS_GEN_USD_PER_CAND", "0.01"))

# Distinct optimization angles → diverse kernels → a meaningful survivor rate.
STRATEGY_HINTS = [
    "One program per (batch, spatial-column) tile of width BLOCK_R; reduce over the full "
    "feature axis in registers, then normalize and write back in the same kernel.",
    "One program per row (batch element); load the whole feature vector with a power-of-2 "
    "BLOCK_F and masking; use tl.sum for the reduction.",
    "Two separate kernels: a reduction kernel computing inverse-RMS per (batch,spatial), then "
    "a normalize kernel — fewer registers per kernel.",
    "Vectorize aggressively over the spatial dimension; one program handles many spatial "
    "positions for one batch element.",
    "Use triton.autotune over BLOCK sizes and num_warps to pick the fastest config.",
    "Tile both the feature and spatial axes with a 2D block in a single fused pass.",
    "Minimize global memory traffic: read each element exactly once, keep the reduction "
    "in-register, write each output exactly once.",
    "Use a large BLOCK_R (e.g. 512) to amortize launch overhead across many spatial columns.",
    "Reduce a (BLOCK_F, BLOCK_R) tile with tl.sum over axis 0 (masked); broadcast rms back.",
    "Keep the accumulation in fp32 for numerical safety even for lower-precision inputs; fuse "
    "normalization into the same kernel.",
]


@dataclass
class SwarmReport:
    mission_id: str
    requested: int
    proposed: int
    survivors: int
    survivor_rate: float
    by_verdict: dict
    total_verify_seconds: float
    cost: dict
    ledger_path: str
    results: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# PROPOSE — N diverse candidates in parallel
# --------------------------------------------------------------------------- #
async def propose_swarm(mission_id: str, n: int, model: str, concurrency: int) -> list[tuple]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(idx: int):
        async with sem:
            claim = Claim(
                mission_id=mission_id,
                statement="A Triton RMSNorm kernel (reduce over dim=1) faster than the PyTorch reference.",
                claim_type="speedup_claim",
                target="36_RMSNorm",
                speedup_threshold=1.0,
            )
            hint = STRATEGY_HINTS[idx % len(STRATEGY_HINTS)]
            try:
                candidate, _ = await propose_candidate(
                    claim, model=model, extra_directive=hint,
                    label=f"swarm_{idx:02d}", meta_extra={"swarm_index": idx},
                )
                # No disk persistence: the oracle verifies candidate.code inline
                # (KernelOracle reads .code first), so N files/run would be pure clutter.
                return (claim, candidate)
            except Exception as e:
                print(f"[swarm] propose #{idx} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                return None

    print(f"[swarm] proposing {n} candidates (gpt-5.4-mini, ≤{concurrency} concurrent)…", file=sys.stderr)
    items = await asyncio.gather(*[_one(i) for i in range(n)])
    return [it for it in items if it is not None]


# --------------------------------------------------------------------------- #
# VERIFY — each candidate through its OWN Orchestrator (shared mission_id),
# concurrent via worker threads so Modal .remote fans out across T4 containers.
# --------------------------------------------------------------------------- #
def _verify_one_sync(claim: Claim, candidate, mission_id: str, idx: int, ledger_path: Optional[str]) -> dict:
    from crucible.ledger import Ledger
    from crucible.oracle.base import CitationOracleAdapter, OracleRouter
    from crucible.oracle.kernel_oracle import KernelOracle
    from crucible.orchestrator import Orchestrator

    t0 = time.monotonic()
    base = {"idx": idx, "candidate_id": candidate.candidate_id, "label": candidate.label,
            "strategy": candidate.strategy,
            "claimed_speedup": (candidate.metadata or {}).get("claimed_speedup")}
    try:
        ledger = Ledger(ledger_path) if ledger_path else Ledger()  # Ledger() = canonical path
        try:
            ledger.conn.execute("PRAGMA busy_timeout=8000")  # tolerate concurrent writers (WAL)
        except Exception:
            pass
        router = OracleRouter(default=KernelOracle()).register("existence_claim", CitationOracleAdapter())
        orch = Orchestrator(
            oracle=router, ledger=ledger, mission_id=mission_id, annotate=False,
            user_id="veritas-swarm", convo_id="autoresearch-hackathon",
        )
        outcome = orch.evaluate(claim, candidate, mission_name=f"swarm[{idx}]:{candidate.label}")
        return {
            **base, "status": "ok",
            "verdict": outcome.verdict.verdict, "promoted": bool(outcome.promoted),
            "promotion": outcome.promotion, "speedup": getattr(outcome, "speedup", None),
            "ledger_id": getattr(outcome, "ledger_id", None),
            "trace_id": getattr(outcome, "trace_id", None),
            "blocked_reason": outcome.blocked_reason, "seconds": time.monotonic() - t0,
        }
    except Exception as e:
        print(f"[swarm] verify #{idx} ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return {**base, "status": "error", "verdict": None, "promoted": False,
                "promotion": "blocked", "error": f"{type(e).__name__}: {e}",
                "seconds": time.monotonic() - t0}


async def verify_swarm(items: list[tuple], mission_id: str, concurrency: int,
                       ledger_path: Optional[str]) -> list[dict]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _verify(idx: int, claim: Claim, candidate):
        async with sem:
            return await asyncio.to_thread(_verify_one_sync, claim, candidate, mission_id, idx, ledger_path)

    print(f"[swarm] verifying {len(items)} candidates through the REAL gate "
          f"(≤{concurrency} concurrent Modal T4 containers)…", file=sys.stderr)
    return await asyncio.gather(*[_verify(i, c, cand) for i, (c, cand) in enumerate(items)])


# --------------------------------------------------------------------------- #
def _cost_estimate(total_verify_seconds: float, proposed: int) -> dict:
    gpu = total_verify_seconds * T4_USD_PER_SEC
    gen = proposed * GEN_USD_PER_CANDIDATE
    return {
        "t4_usd_per_sec": T4_USD_PER_SEC,
        "verify_wallclock_seconds": round(total_verify_seconds, 1),
        "est_gpu_usd_upper_bound": round(gpu, 4),
        "est_generation_usd": round(gen, 4),
        "est_total_usd": round(gpu + gen, 4),
        "note": ("GPU figure is an UPPER BOUND — verify wall-clock includes Modal container "
                 "cold-start + client overhead, not only billed GPU-seconds. Generation cost is "
                 "a rough gpt-5.4-mini per-candidate estimate."),
    }


def _build_report(mission_id, requested, items, results, ledger_path) -> SwarmReport:
    from crucible.ledger import DEFAULT_LEDGER_PATH
    proposed = len(items)
    # Attach source code to SURVIVOR results so they can feed crucible-core's GPU
    # curve (run_curve ladder) — survivors only, to keep the report light.
    code_map = {c.candidate_id: c.code for (_c, c) in items}
    for r in results:
        if r.get("promoted"):
            r["code"] = code_map.get(r.get("candidate_id"))
    survivors = sum(1 for r in results if r.get("promoted"))
    by_verdict = Counter((r.get("verdict") if r.get("status") == "ok" else "error") or "unknown"
                         for r in results)
    total_secs = sum(r.get("seconds", 0.0) for r in results if r.get("status") == "ok")
    return SwarmReport(
        mission_id=mission_id, requested=requested, proposed=proposed,
        survivors=survivors, survivor_rate=(survivors / proposed if proposed else 0.0),
        by_verdict=dict(by_verdict), total_verify_seconds=total_secs,
        cost=_cost_estimate(total_secs, proposed),
        ledger_path=ledger_path or DEFAULT_LEDGER_PATH, results=results,
    )


async def run_swarm(*, n: int, model: str, propose_concurrency: int,
                    verify_concurrency: int, ledger_path: Optional[str]) -> SwarmReport:
    from crucible.raindrop_bridge import install_raindrop_bridge

    mission_id = new_id("swarm")
    print(f"[swarm] mission_id={mission_id}", file=sys.stderr)
    bridge = install_raindrop_bridge(user_id="veritas-swarm", convo_id="autoresearch-hackathon")
    try:
        items = await propose_swarm(mission_id, n, model, propose_concurrency)
    finally:
        if bridge is not None:
            bridge.flush()
    if not items:
        return _build_report(mission_id, n, [], [], ledger_path)
    results = await verify_swarm(items, mission_id, verify_concurrency, ledger_path)
    results.sort(key=lambda r: r.get("idx", 0))
    return _build_report(mission_id, n, items, results, ledger_path)


# --------------------------------------------------------------------------- #
def survivor_ladder(report: SwarmReport) -> list[tuple]:
    """Swarm survivors as a STRICTLY-INCREASING-speedup ladder for crucible-core's
    GPU self-improvement curve (run_curve): each rung beats the prior frontier, so
    the 1→2→3 climb is gate-enforceable. Returns ``[(candidate_id, code, speedup)]``
    in ascending measured-speedup order (a greedy strictly-increasing subsequence)."""
    survs = sorted(
        (r for r in report.results
         if r.get("promoted") and isinstance(r.get("speedup"), (int, float)) and r.get("code")),
        key=lambda r: r["speedup"],
    )
    ladder, last = [], 0.0
    for r in survs:
        if r["speedup"] > last + 1e-9:
            ladder.append((r["candidate_id"], r["code"], float(r["speedup"])))
            last = r["speedup"]
    return ladder


def generate_candidate_sources(
    n: int = 10, *, model: str = DEFAULT_MODEL,
    mission_id: Optional[str] = None, concurrency: int = 8,
) -> list[tuple]:
    """Integration helper for modal-oracle's ``live_swarm.run_megastructure``.

    Propose N diverse gpt-5.4-mini candidates and return ``[(candidate_id, code, "")]``
    ready to pass as ``run_megastructure(extra_sources=...)`` (expect="" — the gate
    decides). MUST be called OUTSIDE an event loop (run_megastructure uses Modal's
    sync ``fn.map``, which can't iterate from async)."""
    mid = mission_id or new_id("swarm")
    items = asyncio.run(propose_swarm(mid, n, model, concurrency))
    return [(c.candidate_id, c.code, "") for (_c, c) in items]


def _print_report(rep: SwarmReport) -> None:
    print("\n  VERITAS — VERIFIED SWARM FAN-OUT (the megastructure at demo scale)\n")
    print(f"  mission_id      : {rep.mission_id}   (the swarm courtroom — query crucible.mission_id)")
    print(f"  requested / proposed : {rep.requested} / {rep.proposed}")
    print(f"  SURVIVORS (committed): {rep.survivors}/{rep.proposed}  "
          f"= {rep.survivor_rate*100:.0f}% verified-survivor rate")
    print(f"  verdict breakdown    : {rep.by_verdict}")
    c = rep.cost
    print(f"  cost (estimate)      : ~${c['est_total_usd']} "
          f"(GPU ≤${c['est_gpu_usd_upper_bound']} over {c['verify_wallclock_seconds']}s @ "
          f"${c['t4_usd_per_sec']}/s T4 + gen ~${c['est_generation_usd']})")
    print(f"  ledger               : {rep.ledger_path}")
    print(f"\n  per-candidate (only survivors commit — same gate as every candidate):")
    print(f"    {'#':>2}  {'verdict':10} {'promotion':10} {'speedup':>8}  {'claimed':>7}  detail")
    for r in rep.results:
        sp = r.get("speedup")
        sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else "—"
        cl = r.get("claimed_speedup")
        cl_s = f"{cl:.2f}x" if isinstance(cl, (int, float)) else "—"
        detail = ""
        if r.get("promoted"):
            detail = f"COMMITTED ledger={r.get('ledger_id')}"
        elif r.get("status") == "error":
            detail = f"verifier error: {r.get('error','')[:60]}"
        elif r.get("blocked_reason"):
            detail = f"blocked: {r['blocked_reason'][:60]}"
        print(f"    {r.get('idx'):>2}  {str(r.get('verdict')):10} {str(r.get('promotion')):10} "
              f"{sp_s:>8}  {cl_s:>7}  {detail}")
    print(f"\n  >> {rep.survivors} verified increment(s) entered the compounding ledger; "
          f"the rest were blocked on merit. {c['note']}\n")


def _reset_ledger(path: str) -> None:
    p = pathlib.Path(path)
    removed = []
    for suffix in ("", "-wal", "-shm"):
        f = pathlib.Path(str(p) + suffix)
        if f.exists():
            f.unlink()
            removed.append(f.name)
    print(f"[swarm] RESET ledger: removed {removed or '(none present)'} at {p.parent}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="VERITAS verified swarm fan-out")
    ap.add_argument("--n", type=int, default=10, help="number of candidates to fan out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--propose-concurrency", type=int, default=8)
    ap.add_argument("--verify-concurrency", type=int, default=6,
                    help="concurrent Modal T4 verifications (containers)")
    ap.add_argument("--ledger", default=None,
                    help="ledger db path (default: canonical VERITAS_LEDGER_DB / <repo>/veritas_ledger.db)")
    ap.add_argument("--reset-ledger", action="store_true",
                    help="DELETE the target ledger file before running (clean survivor count)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not set — cannot run the live swarm.", file=sys.stderr)
        return 2

    if args.reset_ledger:
        from crucible.ledger import DEFAULT_LEDGER_PATH
        _reset_ledger(args.ledger or DEFAULT_LEDGER_PATH)

    try:
        rep = asyncio.run(run_swarm(
            n=args.n, model=args.model,
            propose_concurrency=args.propose_concurrency,
            verify_concurrency=args.verify_concurrency,
            ledger_path=args.ledger,
        ))
    except Exception:
        print("swarm FAILED:\n" + traceback.format_exc(), file=sys.stderr)
        return 1

    if args.json:
        import json
        print(json.dumps({
            "mission_id": rep.mission_id, "requested": rep.requested, "proposed": rep.proposed,
            "survivors": rep.survivors, "survivor_rate": rep.survivor_rate,
            "by_verdict": rep.by_verdict, "cost": rep.cost, "ledger_path": rep.ledger_path,
            "results": rep.results,
        }, indent=2, default=str))
    else:
        _print_report(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
