"""VERITAS live megastructure swarm — concurrent Modal fan-out through the REAL gate (Task #10).

The ceiling beat: N candidates fan out across M real Modal T4 sandboxes CONCURRENTLY (the
megastructure made real — proven by M distinct MODAL_TASK_IDs), every verdict GATE-PRODUCED
through crucible-core's Orchestrator (cheats blocked, the honest increment committed), then
run#2 compounds on run#1's verified ledger row. The honest 2.41x is a real measured win; the
stream cheat is caught by the dual-timer disagreement (the verdict). Nothing canned.

Design: the slow part (the live Modal verification) is fanned out concurrently FIRST; the
resulting LIVE verdicts are then driven through the real Orchestrator (spans + truth-floor gate
+ ledger) via `FannedOracle` — so the courtroom is gate-produced AND the swarm is concurrent,
without verifying any candidate twice. Disguise/bypass cheats are static-pre-gated client-side
and never spin a sandbox (they die before GPU spend).

Members can come from rmsnorm_lab candidate FILES (by label) or as RAW SOURCE strings (the
openai-generator's live candidates, Task #11) — see `run_megastructure(extra_sources=...)`.

modal-oracle owns this live-oracle flow; demo-verifier wires `run_megastructure()` into
demo.py --live and narrates it.
"""
from __future__ import annotations

import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Optional

from crucible.oracle import static_checker
from crucible.oracle.kernel_oracle import (
    APP_NAME,
    FUNCTION_NAME,
    candidate_source,
    reference_source,
    to_verdict,
)
from crucible.schemas import Candidate, Claim

TARGET = "36_RMSNorm"

# Default swarm: a cheat + the honest increment + two more cheats. All pass the static gate and
# reach Modal (so they spin real sandboxes); their judges are runtime. Callers append generated
# candidates (Task #11) via `extra` (file labels) or `extra_sources` (raw source strings).
DEFAULT_SWARM = [
    ("tamper_stream", "refuted"),       # dual-timer disagreement (the narrated verdict)
    ("good_rehearsed", "confirmed"),    # the real 2.41x BW-bound win -> committed
    ("wrong_lastdim", "refuted"),       # honest mistake, correctness fail
    ("tamper_result_reuse", "refuted"), # materialization / poison+isnan
]

# Internal normalized member: a stable id, a display label, the source string, the expected verdict.
_Member = namedtuple("_Member", "cid label source expect")


@dataclass
class MemberOutcome:
    label: str
    candidate_id: str
    verdict: str
    promoted: bool
    correctness_passed: bool
    tamper_detected: bool
    speedup: Optional[float]
    blocked_reason: Optional[str]
    modal_task_id: Optional[str]      # which Modal container verified it (None if static-pre-gated)
    ledger_id: Optional[str]
    proof_hash: Optional[str]


@dataclass
class MegastructureResult:
    members: list[MemberOutcome]
    n_candidates: int
    n_sandboxes: int                  # distinct MODAL_TASK_IDs (the megastructure width)
    sandbox_ids: list[str]
    fanout_wall_s: float              # wall time of the concurrent fan-out (whole swarm)
    parallelism: int                  # distinct live T4 containers used at once (megastructure width)
    committed: list[str]              # committed ledger_ids
    blocked: list[str]                # blocked labels
    compounding: Optional[dict] = None  # {baseline_ledger_id, parent_ledger_id, promoted, run_id}
    trace_id: Optional[str] = None
    run_url: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


class FannedOracle:
    """Oracle that returns ALREADY-FANNED-OUT live verdicts (keyed by candidate_id) so the
    Orchestrator runs the real gate over them without re-verifying. Satisfies the Oracle
    protocol. The verdicts are genuine live Modal results — the fan-out is just the dispatch."""

    name = "kernel_oracle_fanned"
    oracle_type = "kernel"

    def __init__(self, verdicts_by_cid: dict[str, dict]):
        self._v = verdicts_by_cid

    def verify(self, claim: Claim, candidate: Candidate):
        d = self._v.get(candidate.candidate_id)
        if d is None:
            from crucible.schemas import Verdict
            return Verdict(
                claim_id=claim.claim_id, candidate_id=candidate.candidate_id,
                mission_id=claim.mission_id, verdict="unverified", verifier_status="ERROR",
                error=f"no fanned verdict for {candidate.candidate_id}",
            )
        return to_verdict(d, claim, candidate)


def _cid(label: str) -> str:
    return label if label.startswith("cnd_") else f"cnd_{label}"


def _resolve_members(
    members: list[tuple[str, str]],
    extra: Optional[list[tuple[str, str]]],
    extra_sources: Optional[list[tuple[str, str, str]]],
) -> list[_Member]:
    """Normalize file-label members [(label, expect)] and raw-source members
    [(cid, source, expect)] into a single _Member list."""
    out: list[_Member] = []
    for label, expect in list(members) + list(extra or []):
        out.append(_Member(_cid(label), label, candidate_source(label), expect))
    for cid, source, expect in (extra_sources or []):
        cid = _cid(cid)
        out.append(_Member(cid, cid, source, expect))
    return out


def _payload(m: _Member, spec: Optional[dict]) -> dict:
    p = {
        "reference_src": reference_source(),
        "candidate_src": m.source,
        "claim_id": f"clm_{m.label}",
        "candidate_id": m.cid,
        "backend": "triton",
        "precision": "fp32",
    }
    if spec:
        p.update({k: v for k, v in spec.items() if k in (
            "tolerance", "num_correct_trials", "num_warmup", "num_perf_trials",
            "seed", "dual_timer_threshold", "excessive_speedup_threshold", "run_static")})
    return p


def _blocked_dict(m: _Member, static: dict) -> dict:
    return {
        "verdict": "blocked", "claim_id": f"clm_{m.label}", "candidate_id": m.cid,
        "oracle_type": "kernel", "correctness_passed": False, "speedup": None,
        "tamper_detected": True, "verifier_status": "OK",
        "blocked_reason": f"static pre-gate: {static['blocked_reason']}",
        "hardware": None, "measured_by": "modal-oracle",
        "details": {"static": static, "sandbox": {"modal_task_id": None}}, "error": None,
    }


def fan_out(members: list[_Member], spec: Optional[dict] = None) -> tuple[dict[str, dict], dict]:
    """Static-pre-gate, then CONCURRENTLY verify the survivors on Modal via .map() (proven in
    phase-zero to scale out across distinct containers). Returns
    ({candidate_id: live_verdict_dict}, telemetry). Disguise/bypass cheats are pre-gated
    client-side and never spin a sandbox (they die before GPU spend)."""
    import modal

    verdicts: dict[str, dict] = {}
    modal_batch: list[_Member] = []
    for m in members:
        st = static_checker.static_pregate(m.source, backend="triton")
        if not st["ok"]:
            verdicts[m.cid] = _blocked_dict(m, st)  # no GPU spend
        else:
            modal_batch.append(m)

    payloads = [_payload(m, spec) for m in modal_batch]
    fanout_wall = 0.0
    if payloads:
        fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
        t0 = time.monotonic()
        results = list(fn.map(payloads))          # concurrent fan-out across containers
        fanout_wall = round(time.monotonic() - t0, 2)
        for m, vdict in zip(modal_batch, results):
            verdicts[m.cid] = vdict

    return verdicts, {"fanout_wall_s": fanout_wall, "n_modal": len(payloads)}


def run_megastructure(
    members: Optional[list[tuple[str, str]]] = None,
    *,
    extra: Optional[list[tuple[str, str]]] = None,
    extra_sources: Optional[list[tuple[str, str, str]]] = None,
    compounding: bool = True,
    spec: Optional[dict] = None,
    ledger: Any = None,
    mission_id: str = "msn_megastructure",
    out_dir: str = "certificates",
) -> MegastructureResult:
    """Run the live megastructure beat.

    members:       [(candidate_label, expected_verdict)] from rmsnorm_lab/candidates/ (default swarm)
    extra:         more file-label members to append
    extra_sources: [(candidate_id, source_str, expected)] — RAW source (the generator's live
                   candidates, Task #11); no file needed
    ledger:        a crucible.ledger.Ledger (else a temp one is used)
    Returns a MegastructureResult for demo.py --live to narrate.
    """
    from crucible.ledger import Ledger
    from crucible.orchestrator import Orchestrator

    members_resolved = _resolve_members(list(members or DEFAULT_SWARM), extra, extra_sources)

    if ledger is None:
        import tempfile
        ledger = Ledger(f"{tempfile.mkdtemp(prefix='veritas_mega_')}/ledger.db")

    # 1) CONCURRENT FAN-OUT on Modal (the megastructure).
    verdicts, tele = fan_out(members_resolved, spec)

    # 2) Drive the live verdicts through the REAL gate (spans + truth-floor + ledger).
    orch = Orchestrator(oracle=FannedOracle(verdicts), ledger=ledger, mission_id=mission_id, out_dir=out_dir)
    items = []
    for m in members_resolved:
        claim = Claim(mission_id=mission_id, statement=f"a faster RMSNorm via {m.label}",
                      claim_type="speedup_claim", target=TARGET,
                      speedup_threshold=(spec or {}).get("speedup_threshold", 1.2))
        cand = Candidate(claim_id=claim.claim_id, mission_id=mission_id, candidate_id=m.cid,
                         code=m.source, label=m.label, generator="rehearsed", metadata={"backend": "triton"})
        items.append((claim, cand))
    outcomes = orch.run(items) or orch.outcomes

    # 3) Assemble per-member telemetry.
    out_by_cid = {o.candidate_id: o for o in outcomes}
    members_out: list[MemberOutcome] = []
    sandbox_ids: list[str] = []
    for m in members_resolved:
        o = out_by_cid.get(m.cid)
        vd = verdicts.get(m.cid, {})
        tid = ((vd.get("details") or {}).get("sandbox") or {}).get("modal_task_id")
        if tid:
            sandbox_ids.append(tid)
        members_out.append(MemberOutcome(
            label=m.label, candidate_id=m.cid,
            verdict=(o.verdict.verdict if o else vd.get("verdict", "unverified")),
            promoted=(o.promoted if o else False),
            correctness_passed=(o.verdict.correctness_passed if o else False),
            tamper_detected=(o.verdict.tamper_detected if o else vd.get("tamper_detected", False)),
            speedup=(o.verdict.speedup if o else vd.get("speedup")),
            blocked_reason=(o.blocked_reason if o else vd.get("blocked_reason")),
            modal_task_id=tid,
            ledger_id=(o.ledger_id if o else None),
            proof_hash=(o.proof_hash if o else None),
        ))

    distinct = sorted(set(sandbox_ids))
    result = MegastructureResult(
        members=members_out,
        n_candidates=len(members_resolved),
        n_sandboxes=len(distinct),
        sandbox_ids=distinct,
        fanout_wall_s=tele["fanout_wall_s"],
        parallelism=len(distinct),
        committed=[m.ledger_id for m in members_out if m.promoted and m.ledger_id],
        blocked=[m.label for m in members_out if not m.promoted],
        trace_id=orch.trace_id,
        run_url=orch.run_url,
        warnings=list(orch.warnings),
    )

    # 4) COMPOUNDING — run#2 reads run#1's committed baseline.
    if compounding:
        baseline = ledger.latest_baseline(TARGET)
        if baseline is not None:
            m2 = _Member(_cid("good_rehearsed"), "good_rehearsed", candidate_source("good_rehearsed"), "confirmed")
            v2, _ = fan_out([m2], spec)
            orch2 = Orchestrator(oracle=FannedOracle(v2), ledger=ledger, mission_id=mission_id, out_dir=out_dir)
            claim2 = Claim(mission_id=mission_id, statement="a further-improved RMSNorm (run#2)",
                           claim_type="speedup_claim", target=TARGET,
                           speedup_threshold=(spec or {}).get("speedup_threshold", 1.2))
            cand2 = Candidate(claim_id=claim2.claim_id, mission_id=mission_id, candidate_id=m2.cid,
                              code=m2.source, label=m2.label, generator="rehearsed", metadata={"backend": "triton"})
            o2 = (orch2.run([(claim2, cand2)]) or orch2.outcomes)[0]
            result.compounding = {
                "baseline_ledger_id": baseline.ledger_id,
                "baseline_speedup": baseline.speedup,
                "parent_ledger_id": o2.ledger_row.parent_ledger_id,
                "promoted": o2.promoted,
                "run_id": orch2.run_id,
                "compounds": bool(o2.ledger_row.parent_ledger_id == baseline.ledger_id),
            }

    return result


if __name__ == "__main__":
    r = run_megastructure()
    print(f"\nMEGASTRUCTURE: {r.n_candidates} candidates -> {r.n_sandboxes} live Modal T4 sandboxes "
          f"(concurrent), whole swarm verified in {r.fanout_wall_s}s")
    for m in r.members:
        print(f"  {m.label:<20} {m.verdict:<10} promoted={m.promoted!s:<5} speedup={m.speedup} "
              f"task={m.modal_task_id} :: {m.blocked_reason or 'committed'}")
    if r.compounding:
        print(f"  compounding: run#2 parent={r.compounding['parent_ledger_id']} "
              f"compounds_on_run1={r.compounding['compounds']}")
