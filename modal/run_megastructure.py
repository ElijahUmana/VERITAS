#!/usr/bin/env python
"""PROOF: the live megastructure beat (Task #10) — N candidates fan out across M real Modal T4
sandboxes concurrently, every verdict gate-produced, cheat caught live, honest committed, run#2
compounds. Asserts the WOW is real (exit 0 iff so). Re-uses crucible.live_swarm.run_megastructure.

    .venv/bin/python modal/run_megastructure.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from crucible.live_swarm import run_megastructure  # noqa: E402


def main() -> int:
    r = run_megastructure()
    by = {m.label: m for m in r.members}

    print("\n" + "=" * 86)
    print(f"LIVE MEGASTRUCTURE — {r.n_candidates} candidates across {r.n_sandboxes} distinct live "
          f"Modal T4 sandboxes (concurrent), whole swarm in {r.fanout_wall_s}s")
    print("=" * 86)
    for m in r.members:
        sp = f"{m.speedup:.2f}x" if isinstance(m.speedup, (int, float)) else "-"
        tag = "COMMIT ✅" if m.promoted else f"BLOCK ❌ ({m.blocked_reason})"
        print(f"  {m.label:<20} {m.verdict:<10} sandbox={m.modal_task_id or '(pre-gated)':<24} "
              f"speedup={sp:<7} {tag}")
    if r.compounding:
        c = r.compounding
        print(f"  compounding: run#2 baseline={c['baseline_ledger_id']} parent={c['parent_ledger_id']} "
              f"compounds={c['compounds']}")
    print(f"  sandboxes: {r.sandbox_ids}")
    print(f"  trace_id: {r.trace_id}")

    stream, honest = by.get("tamper_stream"), by.get("good_rehearsed")
    checks = {
        "megastructure scaled out (>=2 concurrent live sandboxes)": r.n_sandboxes >= 2,
        "cheat caught by its NAMED defense (dual-timer, not co-tenant contamination)":
            bool(stream and stream.blocked_reason and "dual-timer" in stream.blocked_reason),
        "cheat caught live (stream blocked + tamper)": bool(stream and not stream.promoted and stream.tamper_detected),
        "honest committed (proof_hash + real speedup)": bool(honest and honest.promoted and honest.proof_hash and honest.speedup and honest.speedup > 1.2),
        "honest speedup believable (<5x, not a reward-hack)": bool(honest and honest.speedup and honest.speedup < 5.0),
        "all non-honest blocked": all(not m.promoted for m in r.members if m.label != "good_rehearsed"),
        "run#2 compounds on run#1": bool(r.compounding and r.compounding["compounds"]),
    }
    print("\n" + "-" * 86)
    ok = True
    for k, v in checks.items():
        ok = ok and v
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("-" * 86)
    print("RESULT:", "✅ LIVE MEGASTRUCTURE VERIFIED — real GPU swarm, gate-produced, cheat caught, compounding."
          if ok else "❌ megastructure check failed (real finding).")
    print(f"(MCP-verify gate-produced: query Workshop run {r.trace_id} for spans with crucible.oracle_type set.)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
