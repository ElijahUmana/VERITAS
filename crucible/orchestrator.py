"""crucible/orchestrator.py — the CRUCIBLE engine (FLOOR.md §2).

For each (Claim, Candidate) it runs the full courtroom motion:

    assign IDs
      → emit claim / candidate / verify spans   (raindrop-courtroom's CrucibleTracer)
      → dispatch to the Oracle  (modal-oracle's kernel oracle / citation oracle / ...)
      → emit oracle + anti_tamper spans, FLUSH        (phase 1)
      → read the trace BACK from Workshop: does the claim have an oracle span?
        (detector D == FLOOR §2.3 `trace_readback_confirmed`)
      → gate on the §2.3 truth floor  (crucible.schemas.evaluate_truth_floor)
      → promote: append the verified ledger row + write the Claim Certificate
        OR block: append a negative-evidence row
      → emit the ledger span, FLUSH                   (phase 2)
      → adjudicate + annotate the run                 (the courtroom audit)
      → trigger replay                                (re-verify the increment)

The gate input is read back from Workshop (not assumed) — a claim is promotable
only if its oracle span actually LANDED, which is the whole point of §2.3.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from crucible.certificate import build_certificate, write_certificate
from crucible.detectors import (
    adjudicate,
    judge_and_annotate,
    trace_readback_confirmed as _detector_readback,
)
from crucible.ledger import Ledger
from crucible.oracle.base import Oracle, make_error_verdict
from crucible.schemas import (
    Candidate,
    Certificate,
    Claim,
    GateResult,
    LedgerRow,
    Verdict,
    canonical_hash,
    evaluate_truth_floor,
    new_id,
)
from crucible.trace import CrucibleTracer


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@dataclass
class ClaimOutcome:
    claim: Claim
    candidate: Candidate
    verdict: Verdict
    gate: GateResult
    ledger_row: LedgerRow
    trace_readback_confirmed: bool
    certificate: Optional[Certificate] = None
    certificate_paths: Optional[tuple[Path, Path]] = None
    warnings: list[str] = field(default_factory=list)

    # --- flat accessors (the contract demo-verifier / generator read) ---
    @property
    def promoted(self) -> bool:
        return self.gate.promoted

    @property
    def claim_id(self) -> str:
        return self.claim.claim_id

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def mission_id(self) -> str:
        return self.claim.mission_id

    @property
    def trace_id(self) -> str:
        return self.ledger_row.trace_id

    @property
    def ledger_id(self) -> str:
        return self.ledger_row.ledger_id

    @property
    def proof_hash(self) -> str:
        return self.ledger_row.proof_hash

    @property
    def promotion(self) -> str:
        return self.gate.promotion

    @property
    def blocked_reason(self) -> Optional[str]:
        return self.gate.blocked_reason

    @property
    def certificate_id(self) -> Optional[str]:
        return self.certificate.certificate_id if self.certificate else None

    @property
    def speedup(self) -> Optional[float]:
        return self.verdict.speedup


class Orchestrator:
    """Drives claims through the CRUCIBLE gate for one mission (one Workshop run)."""

    def __init__(
        self,
        *,
        oracle: Oracle,
        ledger: Ledger,
        mission_id: Optional[str] = None,
        run_id: Optional[int] = None,
        tracer: Optional[CrucibleTracer] = None,
        out_dir: str | Path = "certificates",
        replay_url: Optional[str] = None,
        annotate: bool = True,
        event_name: str = "veritas_crucible",
        user_id: str = "veritas-crucible",
        convo_id: str = "autoresearch-hackathon",
        base: Optional[str] = None,
    ):
        self.oracle = oracle
        self.ledger = ledger
        self.mission_id = mission_id or new_id("msn")
        self.run_id = run_id if run_id is not None else ledger.next_run_id()
        self.tracer = tracer or CrucibleTracer(
            self.mission_id, event_name=event_name, user_id=user_id, convo_id=convo_id, base=base,
        )
        self.out_dir = Path(out_dir)
        self.replay_url = replay_url.rstrip("/") if replay_url else None
        self.annotate = annotate
        self.outcomes: list[ClaimOutcome] = []
        self.warnings: list[str] = []
        self._registry: dict[tuple[str, str], tuple[Claim, Candidate]] = {}
        self._mission_span = None

    # --------------------------------------------------------------------- #
    @property
    def trace_id(self) -> str:
        return self.tracer.trace_id

    @property
    def run_url(self) -> str:
        return self.tracer.run_url

    @property
    def mission_span_id(self) -> Optional[str]:
        """The current mission span's id (set during run()) — lets a caller emit
        sibling spans into the same run (e.g. adversarial detector-probe injections)."""
        return self._mission_span.span_id if self._mission_span is not None else None

    def _oracle_type_hint(self, claim: Claim) -> str:
        return "citation" if claim.claim_type == "existence_claim" else "kernel"

    def query(self, sql: str, *, limit: int = 50, max_bytes: int = 200000) -> dict:
        """Read-only SQL against the local Workshop daemon (same surface the
        raindrop MCP ``query_traces`` and detectors use)."""
        payload = json.dumps({"sql": sql, "limit": limit, "max_bytes": max_bytes}).encode()
        req = urllib.request.Request(
            f"{self.tracer.base}/api/traces/query", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    # --- the gate input, READ BACK from Workshop (FLOOR §2.3 / detector D) -- #
    def _has_oracle_span(self, claim_id: str) -> bool:
        sql = (
            "SELECT COUNT(*) AS n FROM spans s "
            f"WHERE s.run_id={_sql_literal(self.tracer.trace_id)} "
            f"AND json_extract(s.attributes,'$.\"crucible.claim_id\"')={_sql_literal(claim_id)} "
            "AND json_extract(s.attributes,'$.\"crucible.oracle_type\"') IS NOT NULL"
        )
        data = self.query(sql, limit=1)
        rows = data.get("rows", []) if isinstance(data, dict) else []
        if rows:
            row = rows[0]
            try:
                return int(row.get("n") or row.get("COUNT(*)") or 0) > 0
            except (TypeError, ValueError):
                return False
        return False

    def _trace_readback_confirmed(self, claim_id: str, *, retries: int = 6, delay: float = 0.4) -> bool:
        """§2.3: confirmed iff the claim has a landed oracle span in Workshop.

        Polls (ingest→SQLite can lag the POST), fails SAFE (False) on transport
        error — a claim with no verifiable oracle span must NOT promote.
        """
        last_err: Optional[str] = None
        for _ in range(max(1, retries)):
            try:
                # raindrop-courtroom's canonical detector-D helper = single source of truth
                if _detector_readback(self.tracer.trace_id, claim_id, base=self.tracer.base):
                    return True
            except Exception as exc:  # transport / query error — try our own query, keep polling
                last_err = f"{type(exc).__name__}: {exc}"
                try:
                    if self._has_oracle_span(claim_id):
                        return True
                except Exception:
                    pass
            time.sleep(delay)
        if last_err:
            self.warnings.append(f"trace readback for {claim_id} unconfirmed (last error: {last_err})")
        return False

    # --- per-claim courtroom motion --------------------------------------- #
    def _verify_one(self, claim: Claim, candidate: Candidate) -> ClaimOutcome:
        warnings: list[str] = []
        self._registry[(claim.claim_id, candidate.candidate_id)] = (claim, candidate)

        # claim span stays OPEN as the parent until the end of this claim
        cspan = self.tracer.span(
            "claim", "agent_root", f"claim: {claim.statement[:70]}",
            claim_id=claim.claim_id, parent=self._mission_span,
            input=claim.statement, crucible_claim_type=claim.claim_type,
        )

        # candidate proposal span (finished + flushed in phase 1 -> claim_id visible)
        candspan = self.tracer.span(
            "candidate", "llm_call", f"candidate: {candidate.label or candidate.candidate_id}",
            claim_id=claim.claim_id, candidate_id=candidate.candidate_id, parent=cspan,
            model=candidate.generator,
        )
        candspan.finish(
            status="OK",
            input=(candidate.strategy or "(candidate artifact)"),
            output=f"artifact_hash={candidate.artifact_hash}",
        )

        # verify span wraps the oracle dispatch
        vspan = self.tracer.span(
            "verify", "tool_call", "verify (oracle dispatch)",
            claim_id=claim.claim_id, candidate_id=candidate.candidate_id, parent=cspan,
        )
        try:
            verdict = self.oracle.verify(claim, candidate)
        except Exception as exc:  # the VERIFIER broke — ERROR verdict, surfaced loudly
            verdict = make_error_verdict(claim, candidate, exc, oracle_type=self._oracle_type_hint(claim))
            warnings.append(f"oracle raised: {verdict.error}")
        v_status = "ERROR" if verdict.verifier_status == "ERROR" else "OK"
        vspan.finish(
            status=v_status, verdict=verdict.verdict,
            correctness_passed=verdict.correctness_passed, tamper_detected=verdict.tamper_detected,
            speedup=verdict.speedup, blocked_reason=verdict.blocked_reason,
            input=claim.statement, output=(verdict.error or verdict.blocked_reason or verdict.verdict),
        )

        # oracle span (granular oracle_type: keeps FLOOR §2.1 span enum intact)
        primary = "citation" if verdict.oracle_type == "citation" else "correctness"
        ospan = self.tracer.span(
            "oracle", "tool_call", f"oracle: {verdict.oracle_type}",
            claim_id=claim.claim_id, candidate_id=candidate.candidate_id,
            oracle_type=primary, parent=vspan,
        )
        ospan.finish(
            status=v_status, verdict=verdict.verdict,
            correctness_passed=verdict.correctness_passed, speedup=verdict.speedup,
            blocked_reason=verdict.blocked_reason,
            output=self._verdict_summary(verdict),
        )

        # anti-tamper span (execution oracles only; citation has no tamper surface)
        if verdict.oracle_type != "citation":
            atspan = self.tracer.span(
                "anti_tamper", "tool_call", "anti-tamper checks",
                claim_id=claim.claim_id, candidate_id=candidate.candidate_id,
                oracle_type="anti_tamper", parent=vspan,
            )
            atspan.finish(
                status="OK", tamper_detected=verdict.tamper_detected,
                blocked_reason=(verdict.blocked_reason if verdict.tamper_detected else None),
                output=("tamper detected — reward-hack blocked" if verdict.tamper_detected else "no tamper detected"),
            )

        # PHASE 1 FLUSH — land the spans the gate must read back
        self.tracer.flush()

        # READ BACK -> the §2.3 trace_readback_confirmed input
        trace_ok = self._trace_readback_confirmed(claim.claim_id)

        # THE TRUTH FLOOR
        gate = evaluate_truth_floor(claim, verdict, trace_ok)

        # PROMOTE or BLOCK
        certificate: Optional[Certificate] = None
        cert_paths = None
        if gate.promoted:
            baseline = self.ledger.latest_baseline(claim.target)
            certificate = build_certificate(
                claim, candidate, verdict, trace_id=self.tracer.trace_id, run_id=self.run_id,
            )
            row = LedgerRow(
                mission_id=claim.mission_id, claim_id=claim.claim_id, candidate_id=candidate.candidate_id,
                run_id=self.run_id, claim=claim.statement, claim_type=claim.claim_type, target=claim.target,
                artifact_hash=candidate.artifact_hash or "", artifact_path=candidate.source_path,
                verdict=verdict.verdict, promotion="committed", speedup=verdict.speedup,
                baseline_speedup=(baseline.speedup if baseline else None),
                parent_ledger_id=(baseline.ledger_id if baseline else claim.baseline_ledger_id),
                proof_hash=certificate.proof_hash, trace_id=self.tracer.trace_id,
                certificate_id=certificate.certificate_id,
                evidence={"max_abs_err": (verdict.correctness.max_abs_err if verdict.correctness else None)},
            )
            self.ledger.record(row)
            certificate.ledger_id = row.ledger_id
            cert_paths = write_certificate(certificate, self.out_dir)
        else:
            row = LedgerRow(
                mission_id=claim.mission_id, claim_id=claim.claim_id, candidate_id=candidate.candidate_id,
                run_id=self.run_id, claim=claim.statement, claim_type=claim.claim_type, target=claim.target,
                artifact_hash=candidate.artifact_hash or "", artifact_path=candidate.source_path,
                verdict=verdict.verdict, promotion="blocked", speedup=verdict.speedup,
                parent_ledger_id=claim.baseline_ledger_id,
                proof_hash=canonical_hash({
                    "artifact_hash": candidate.artifact_hash, "verdict": verdict.verdict,
                    "blocked_reason": gate.blocked_reason,
                }),
                trace_id=self.tracer.trace_id, certificate_id=None,
                blocked_reason=gate.blocked_reason,
                evidence={"failed_conditions": gate.failed_conditions},
            )
            self.ledger.record(row)

        # ledger span (detector B audits this: committed must imply verdict==confirmed)
        lspan = self.tracer.span(
            "ledger", "tool_call", f"ledger: {gate.promotion}",
            claim_id=claim.claim_id, candidate_id=candidate.candidate_id, parent=cspan,
        )
        lspan.finish(
            status="OK", promotion=gate.promotion, verdict=verdict.verdict, ledger_id=row.ledger_id,
            blocked_reason=(gate.blocked_reason if not gate.promoted else None),
            output=(f"committed proof_hash={row.proof_hash[:16]}" if gate.promoted else f"blocked: {gate.blocked_reason}"),
        )

        # close the claim span
        cspan.finish(
            status="OK", verdict=verdict.verdict, promotion=gate.promotion,
            output=f"{gate.promotion} ({verdict.verdict}); trace_readback={trace_ok}",
        )

        # PHASE 2 FLUSH — land ledger + claim spans (re-sends earlier spans; upsert = idempotent)
        self.tracer.flush()

        return ClaimOutcome(
            claim=claim, candidate=candidate, verdict=verdict, gate=gate, ledger_row=row,
            trace_readback_confirmed=trace_ok, certificate=certificate, certificate_paths=cert_paths,
            warnings=warnings,
        )

    @staticmethod
    def _verdict_summary(v: Verdict) -> str:
        bits = [f"verdict={v.verdict}", f"correct={v.correctness_passed}", f"tamper={v.tamper_detected}"]
        if v.speedup is not None:
            bits.append(f"speedup={v.speedup:.3f}x")
        if v.blocked_reason:
            bits.append(f"reason={v.blocked_reason}")
        return " ".join(bits)

    # --- public entry points ---------------------------------------------- #
    def run(self, items, *, mission_name: Optional[str] = None) -> list[ClaimOutcome]:
        """Run a list of (Claim, Candidate) pairs as one mission (one run)."""
        if isinstance(items, tuple) and len(items) == 2 and isinstance(items[0], Claim):
            items = [items]
        items = list(items)
        self._mission_span = self.tracer.span(
            "mission", "agent_root", mission_name or f"CRUCIBLE mission {self.mission_id}",
            input=f"run #{self.run_id}: {len(items)} claim(s)",
        )
        outcomes: list[ClaimOutcome] = []
        for claim, candidate in items:
            outcomes.append(self._verify_one(claim, candidate))
        self.outcomes = outcomes

        promoted = sum(o.promoted for o in outcomes)
        self._mission_span.finish(
            status="OK",
            output=f"{promoted}/{len(outcomes)} promoted; run #{self.run_id}",
        )
        self.tracer.flush()

        # the courtroom audit (good/issue annotations) — non-gating. raindrop-courtroom's
        # judge_and_annotate runs all 4 detectors + writes annotations in one call.
        if self.annotate:
            try:
                judge_and_annotate(self.tracer.trace_id, base=self.tracer.base)
            except Exception as exc:
                self.warnings.append(f"judge_and_annotate failed (non-gating): {type(exc).__name__}: {exc}")

        return outcomes

    def run_single(self, claim: Claim, candidate: Candidate, *, mission_name: Optional[str] = None) -> ClaimOutcome:
        return self.run([(claim, candidate)], mission_name=mission_name)[0]

    # canonical single-candidate entry-point (the name generator + demo-verifier pin to).
    def evaluate(self, claim: Claim, candidate: Candidate, *, mission_name: Optional[str] = None) -> ClaimOutcome:
        """Run ONE candidate through the full courtroom: assign IDs → oracle → spans →
        §2.3 truth-floor gate → ledger (commit or block) → certificate → replay-ready.

        Returns a :class:`ClaimOutcome`.  ``.verdict`` is the :class:`Verdict`;
        ``.promoted`` / ``.promotion`` / ``.ledger_id`` / ``.proof_hash`` / ``.trace_id`` /
        ``.certificate_id`` / ``.blocked_reason`` are flat accessors."""
        return self.run_single(claim, candidate, mission_name=mission_name)

    # aliases for callers probing alternate names
    evaluate_candidate = evaluate

    def verify_candidate(self, claim: Claim, candidate: Candidate, *, mission_name: Optional[str] = None) -> Verdict:
        """Same full gate, but returns just the :class:`Verdict` (propose→verdict loop)."""
        return self.evaluate(claim, candidate, mission_name=mission_name).verdict

    # --- replay (FLOOR §1 52-60s) ----------------------------------------- #
    def trigger_replay(self, claim_id: str, candidate_id: str) -> dict:
        """Re-verify a committed increment.  POSTs to raindrop-courtroom's replay
        server if configured; otherwise runs an in-process re-verification and
        emits a replay span (promotion=replayed, or regressed if the verdict flips)."""
        if self.replay_url:
            # sourceRunId lets raindrop-courtroom's replay server compute verdict_changed
            # against the original ledger/oracle verdict.
            body = json.dumps({
                "claim_id": claim_id, "candidate_id": candidate_id,
                "sourceRunId": self.tracer.trace_id, "trace_id": self.tracer.trace_id,
            }).encode()
            req = urllib.request.Request(
                f"{self.replay_url}/replay", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read().decode())
            except urllib.error.URLError as exc:
                raise RuntimeError(f"replay server POST failed at {self.replay_url}/replay: {exc}") from exc
        return self._inprocess_replay(claim_id, candidate_id)

    def _inprocess_replay(self, claim_id: str, candidate_id: str) -> dict:
        key = (claim_id, candidate_id)
        if key not in self._registry:
            raise KeyError(f"no registered claim/candidate for replay: {key}")
        claim, candidate = self._registry[key]
        prior = self.ledger.by_candidate(candidate_id)
        verdict = self.oracle.verify(claim, candidate)
        regressed = bool(prior and prior.is_committed and verdict.verdict != "confirmed")
        promotion = "regressed" if regressed else "replayed"

        rspan = self.tracer.span(
            "replay", "tool_call", "replay re-verify",
            claim_id=claim_id, candidate_id=candidate_id, oracle_type="replay",
            parent=self._mission_span,
        )
        rspan.finish(
            status="OK", verdict=verdict.verdict, promotion=promotion,
            correctness_passed=verdict.correctness_passed, speedup=verdict.speedup,
            output=(f"re-verified: {verdict.verdict}"
                    + (f" (was {prior.verdict})" if prior else "")
                    + (" — REGRESSION" if regressed else "")),
        )
        self.tracer.flush()
        return {
            "claim_id": claim_id, "candidate_id": candidate_id,
            "verdict": verdict.verdict, "promotion": promotion, "regressed": regressed,
            "prior_verdict": (prior.verdict if prior else None),
            "trace_id": self.tracer.trace_id,
        }

    def adjudicate(self) -> dict:
        """Run the full 4-detector courtroom report for this mission's run."""
        return adjudicate(self.tracer.trace_id, base=self.tracer.base)


__all__ = ["Orchestrator", "ClaimOutcome"]
