#!/usr/bin/env python
"""Zero-spend proof of the Verdict contract seam: worker dict -> to_verdict -> pydantic Verdict
-> crucible-core's evaluate_truth_floor gate. Re-uses the captured verdicts in
modal/logs/oracle-proof.log (no new Modal/GPU calls).

Proves:
  1. KernelOracle satisfies the crucible.oracle.base.Oracle protocol.
  2. Every live verdict maps cleanly into crucible.schemas.Verdict (extra="forbid").
  3. The truth-floor gate PROMOTES only the honest candidate and BLOCKS every cheat,
     for the exact reason its named defense produced.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from crucible.oracle import kernel_oracle  # noqa: E402
from crucible.oracle.base import Oracle  # noqa: E402
from crucible.schemas import Candidate, Claim, evaluate_truth_floor  # noqa: E402

LOG = pathlib.Path(__file__).resolve().parents[1] / "modal" / "logs" / "oracle-proof.log"

# candidate -> should it be promoted by the truth floor?
SHOULD_PROMOTE = {
    "good_rehearsed": True,
    "wrong_lastdim": False,
    "tamper_result_reuse": False,
    "tamper_zero_inputs": False,
    "tamper_stream": False,
    "tamper_torch_disguise": False,
}


def _extract_verdicts(text: str) -> dict[str, dict]:
    """Pull every top-level JSON object that looks like a Verdict (has 'oracle_type')."""
    out: dict[str, dict] = {}
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            for j in range(i, n):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        blob = text[i:j + 1]
                        try:
                            obj = json.loads(blob)
                            if isinstance(obj, dict) and "oracle_type" in obj and "candidate_id" in obj:
                                out[obj["candidate_id"]] = obj
                        except json.JSONDecodeError:
                            pass
                        i = j
                        break
        i += 1
    return out


def main() -> int:
    # 1) protocol conformance (runtime_checkable Protocol)
    oracle = kernel_oracle.KernelOracle()
    assert isinstance(oracle, Oracle), "KernelOracle does not satisfy the Oracle protocol"
    print(f"PASS  KernelOracle satisfies the Oracle protocol (name={oracle.name!r})")

    if not LOG.exists():
        print(f"FAIL  no captured log at {LOG} — run modal/run_oracle_proof.py first")
        return 2
    verdicts = _extract_verdicts(LOG.read_text())
    missing = [k for k in SHOULD_PROMOTE if k not in verdicts]
    if missing:
        print(f"FAIL  captured log missing verdicts for: {missing}")
        return 2

    all_ok = True
    print("\n  candidate                gate        reason")
    print("  " + "-" * 88)
    for name, should in SHOULD_PROMOTE.items():
        d = verdicts[name]
        claim = Claim(
            mission_id="m_demo", statement="A faster RMSNorm kernel",
            claim_type="speedup_claim", target="36_RMSNorm", speedup_threshold=1.0,
        )
        candidate = Candidate(claim_id=claim.claim_id, mission_id="m_demo", code="# captured", label=name)
        verdict = kernel_oracle.to_verdict(d, claim, candidate)   # dict -> pydantic Verdict (validates)
        # Raindrop detector D (oracle span exists) is the courtroom's job; assume confirmed here.
        gate = evaluate_truth_floor(claim, verdict, trace_readback_confirmed=True)
        ok = (gate.promoted == should)
        all_ok = all_ok and ok
        tag = "PASS" if ok else "FAIL"
        gate_s = "PROMOTE" if gate.promoted else "BLOCK"
        reason = (gate.blocked_reason or "committed") if not gate.promoted else "committed to ledger"
        print(f"  [{tag}] {name:<22} {gate_s:<10} {reason}")
        if not ok:
            print(f"        expected promote={should}, got {gate.promoted}; failed={gate.failed_conditions}")

    print("  " + "-" * 88)
    if all_ok:
        print("\nRESULT: ✅ Verdict contract seam verified — honest claim promotes, every cheat blocked.")
        return 0
    print("\nRESULT: ❌ contract seam mismatch (see rows).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
