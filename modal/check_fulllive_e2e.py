#!/usr/bin/env python
"""Full-LIVE end-to-end de-risk: the REAL gate centerpiece + live compounding, driven through
crucible-core's Orchestrator + Ledger with the live Modal KernelOracle. No canned anything.

Proves the path the demo's centerpiece (#8) + compounding (#6) need:
  RUN 1  cheat (tamper_stream) -> live Modal oracle -> REFUTED -> gate BLOCKS (negative evidence)
         honest (good_rehearsed) -> live Modal oracle -> CONFIRMED -> trace-readback -> COMMITTED (proof_hash)
  RUN 2  honest increment reads RUN 1's committed ledger row as baseline (parent_ledger_id) -> COMPOUNDING

Isolated: temp ledger + its own mission, so it never touches demo-verifier's demo state.
Requires: `modal deploy modal/verifier_app.py` (done) + the local Workshop daemon up (:5899).

Usage:  .venv/bin/python modal/check_fulllive_e2e.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from crucible.ledger import Ledger  # noqa: E402
from crucible.oracle.kernel_oracle import KernelOracle, candidate_source  # noqa: E402
from crucible.orchestrator import Orchestrator  # noqa: E402
from crucible.schemas import Candidate, Claim  # noqa: E402

TARGET = "36_RMSNorm"


def mk(label: str, mission_id: str, threshold: float = 1.2):
    claim = Claim(mission_id=mission_id, statement=f"a faster RMSNorm via {label}",
                  claim_type="speedup_claim", target=TARGET, speedup_threshold=threshold)
    cand = Candidate(claim_id=claim.claim_id, mission_id=mission_id, code=candidate_source(label),
                     label=label, generator="rehearsed", metadata={"backend": "triton"})
    return claim, cand


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="veritas_oracle_e2e_"))
    ledger = Ledger(str(tmp / "ledger.db"))
    mission = "msn_oracle_e2e"
    print(f"isolated ledger={tmp/'ledger.db'}  mission={mission}\n")

    # ---- RUN 1: a cheat (blocked) + the honest increment (committed) ----
    orch1 = Orchestrator(oracle=KernelOracle(), ledger=ledger, mission_id=mission, out_dir=str(tmp / "certs"))
    print(f"RUN 1 (run_id={orch1.run_id}) — live oracle through the real gate ...")
    out1 = orch1.run([mk("tamper_stream", mission), mk("good_rehearsed", mission)]) or orch1.outcomes
    committed = None
    for o in out1:
        tag = "COMMIT ✅" if o.promoted else f"BLOCK ❌ ({o.blocked_reason})"
        print(f"  {o.candidate.label:<16} verdict={o.verdict.verdict:<10} promoted={o.promoted!s:<5} "
              f"speedup={o.speedup} ledger_id={o.ledger_id} proof={o.proof_hash[:12]} :: {tag}")
        if o.candidate.label == "good_rehearsed":
            committed = o

    # ---- RUN 2: compounding — read RUN 1's committed row as baseline ----
    baseline = ledger.latest_baseline(TARGET)
    orch2 = Orchestrator(oracle=KernelOracle(), ledger=ledger, mission_id=mission, out_dir=str(tmp / "certs"))
    print(f"\nRUN 2 (run_id={orch2.run_id}) — compounding on baseline ledger_id="
          f"{baseline.ledger_id if baseline else None} (speedup={baseline.speedup if baseline else None}) ...")
    claim2, cand2 = mk("good_rehearsed", mission)
    out2 = orch2.run([(claim2, cand2)]) or orch2.outcomes
    o2 = out2[0]
    parent = o2.ledger_row.parent_ledger_id
    print(f"  {o2.candidate.label:<16} verdict={o2.verdict.verdict:<10} promoted={o2.promoted!s:<5} "
          f"parent_ledger_id={parent}")

    # ---- Assertions ----
    cheat = next((o for o in out1 if o.candidate.label == "tamper_stream"), None)
    checks = {
        "cheat blocked (not committed)": bool(cheat and not cheat.promoted and cheat.verdict.verdict == "refuted"),
        "cheat flagged tamper": bool(cheat and cheat.verdict.tamper_detected),
        "honest committed with proof_hash": bool(committed and committed.promoted and committed.proof_hash),
        "honest real speedup >1.2x": bool(committed and committed.speedup and committed.speedup > 1.2),
        "run2 promoted": bool(o2.promoted),
        "run2 compounds on run1 baseline": bool(baseline and parent == baseline.ledger_id),
        "ledger has >=2 committed rows for target": len(ledger.committed_for_target(TARGET)) >= 2,
    }
    print("\n" + "=" * 78)
    print("FULL-LIVE E2E (real oracle -> real gate -> real ledger -> live compounding)")
    print("=" * 78)
    ok = True
    for k, v in checks.items():
        ok = ok and v
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 78)
    print("RESULT:", "✅ FULL-LIVE PATH VERIFIED — centerpiece + compounding are gate-produced, not canned."
          if ok else "❌ full-live path mismatch (real finding).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
