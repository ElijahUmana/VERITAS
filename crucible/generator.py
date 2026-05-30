#!/usr/bin/env python3
"""crucible/generator.py — the candidate-proposing agent (FLOOR §4 + build-order C).

A gpt-5.4-mini agent (OpenAI Agents SDK) PROPOSES a confident kernel candidate as
strict-JSON structured output (the SDK's ``output_type`` drives the Responses API
``text.format={type:json_schema, strict:true}`` path). The proposal becomes a
``crucible.schemas.Candidate``.

THE LOAD-BEARING PROPERTY (FLOOR §1, build-order C): the generated candidate is
NEVER trusted. It is routed through the **same** CRUCIBLE truth floor as a
rehearsed candidate — no shortcut, no "the agent said so". Concretely it must
clear, in order:

  1. STATIC PRE-GATE  (``oracle/static_checker.py`` — runs locally, today):
     a "torch-in-disguise" / try-except / non-Triton candidate is rejected
     BEFORE any GPU spend.
  2. MODAL EXECUTION ORACLE  (correctness 5-seed + dual-timer speed + anti-tamper):
     delegated to the Modal kernel oracle via the orchestrator. Until that lands
     (modal-oracle Task #2 / crucible-core orchestrator), the candidate is left
     ``unverified`` and the gate (:func:`crucible.schemas.evaluate_truth_floor`)
     BLOCKS it — i.e. a generated candidate gets NO free pass; it stays blocked
     until a separate oracle reproduces its claim. This is the honest, no-false-
     pass behaviour; it auto-upgrades to a real CONFIRMED/REFUTED the moment the
     orchestrator + kernel oracle are importable.

The whole agent run is traced into Raindrop Workshop via the native bridge
(``raindrop_bridge.install_raindrop_bridge`` + ``crucible_workflow``), and the
domain ``crucible.*`` spans (candidate/verify/ledger) are emitted via
``crucible/trace.py``.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
import traceback
from dataclasses import dataclass
from typing import Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from crucible.schemas import (  # noqa: E402
    Candidate, Claim, GateResult, Verdict, evaluate_truth_floor, new_id,
)

DEFAULT_MODEL = "gpt-5.4-mini"
GENERATED_DIR = REPO_ROOT / "benchmarks" / "rmsnorm_lab" / "generated"


# --------------------------------------------------------------------------- #
# The LLM's structured proposal (strict JSON schema via Agents SDK output_type).
# --------------------------------------------------------------------------- #
class CandidateProposal(BaseModel):
    """What the generator agent emits (strict structured output)."""

    strategy: str = Field(description="One-line description of the optimization approach")
    rationale: str = Field(description="Why this is BOTH faster and numerically correct")
    claimed_speedup: float = Field(description="Honest estimated speedup vs the PyTorch reference (e.g. 1.8)")
    entry_point: str = Field(default="ModelNew", description="The nn.Module class name the oracle instantiates")
    code: str = Field(description="A self-contained module: imports + @triton.jit kernel(s) + class ModelNew")


GENERATOR_INSTRUCTIONS = """\
You are a GPU kernel-optimization researcher. Propose ONE Triton kernel that \
computes RMSNorm faster than the PyTorch reference, and return it as the \
structured proposal.

HARD INTERFACE CONTRACT (the oracle rejects anything that violates it):
- Define `class ModelNew(nn.Module)` with
  `__init__(self, num_features: int, eps: float = 1e-5)` and
  `forward(self, x: torch.Tensor) -> torch.Tensor`.
- RMSNorm reduces over dim=1 (the feature axis): rms = sqrt(mean(x**2, dim=1) + eps);
  out = x / rms. The eps is INSIDE the sqrt. There is NO learnable weight.
- Input x has shape (B, num_features, *spatial); reduce over dim=1, broadcast back.
- Output must match the reference numerically (fp32 atol=rtol=1e-2 across 5 seeds,
  plus hidden shapes/seeds you will not see).

ANTI-CHEAT (these are statically rejected before any GPU run, so don't try them):
- Do NOT call torch.rms_norm / torch.nn.functional.* / torch.matmul to do the work
  ("torch in disguise"). Use a real `@triton.jit` kernel with `tl.*` ops.
- No try/except fallback to PyTorch, no fake/lazy tensors.
- Your code must be COMPLETE. Never write a bare `pass` statement ANYWHERE (the
  static gate treats any `pass` as an inheritance-bypass cheat and blocks it).
  Implement every branch and method fully; do not leave placeholders or stubs.

Be confident and concrete. Return: strategy, rationale, claimed_speedup, \
entry_point="ModelNew", and the full `code`.
"""


@dataclass
class GenerationResult:
    candidate: Candidate
    proposal: CandidateProposal
    verdict: Verdict
    gate: GateResult
    source_path: Optional[str]
    gate_path: str           # "orchestrator" | "static-blocked" | "kernel-oracle(no-readback)" | "pending-modal-oracle"
    workshop_url: Optional[str] = None
    outcome: Optional[object] = None   # crucible.orchestrator.ClaimOutcome when routed through the orchestrator


# --------------------------------------------------------------------------- #
# 1. Propose (live gpt-5.4-mini, structured output, traced via the bridge)
# --------------------------------------------------------------------------- #
async def propose_candidate(claim: Claim, *, model: str = DEFAULT_MODEL) -> tuple[Candidate, CandidateProposal]:
    from agents import Agent, Runner

    from crucible.raindrop_bridge import crucible_workflow

    agent = Agent(
        name="KernelGenerator",
        model=model,
        instructions=GENERATOR_INSTRUCTIONS,
        output_type=CandidateProposal,
    )
    prompt = (
        f"Propose a faster kernel for this claim: {claim.statement}\n"
        f"Target benchmark: {claim.target}. "
        f"Required speedup threshold: {claim.effective_threshold}x (be honest)."
    )
    with crucible_workflow(
        "kernel_generator",
        node="candidate",
        crucible_meta={"claim_id": claim.claim_id, "mission_id": claim.mission_id,
                       "model": model},
    ):
        result = await Runner.run(agent, prompt)

    proposal: CandidateProposal = result.final_output
    candidate = Candidate(
        claim_id=claim.claim_id,
        mission_id=claim.mission_id,
        code=proposal.code,
        entry_point=proposal.entry_point or "ModelNew",
        generator=model,
        strategy=proposal.strategy,
        label="generated",
        metadata={
            "rationale": proposal.rationale,
            "claimed_speedup": proposal.claimed_speedup,
        },
    )
    return candidate, proposal


def persist_candidate(candidate: Candidate) -> str:
    """Write the generated source to disk so the Modal oracle loads it exactly like
    a rehearsed candidate (the 'verified-or-blocked identically' guarantee)."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{candidate.candidate_id}.py"
    header = (
        f'"""GENERATED candidate — proposed by {candidate.generator}.\n'
        f"candidate_id: {candidate.candidate_id}\n"
        f"claim_id:     {candidate.claim_id}\n"
        f"strategy:     {candidate.strategy}\n"
        f"claimed_speedup: {candidate.metadata.get('claimed_speedup')}\n"
        f"artifact_hash:   {candidate.artifact_hash}\n"
        f"NOT TRUSTED until the CRUCIBLE gate confirms it (no shortcut).\n"
        f'"""\n'
    )
    path.write_text(header + (candidate.code or "") + "\n")
    candidate.source_path = str(path)
    return str(path)


# --------------------------------------------------------------------------- #
# 2. Route through the SAME gate (no trust shortcut)
# --------------------------------------------------------------------------- #
def _run_orchestrator(claim: Claim, candidate: Candidate):
    """PREFERRED path: route through crucible-core's full Orchestrator (oracle
    router -> Modal/citation oracle -> detector-D trace readback -> truth-floor
    gate -> ledger -> certificate). Returns the ClaimOutcome, or None if the
    orchestrator stack isn't importable (then we fall back — never block)."""
    try:
        from crucible.ledger import Ledger
        from crucible.oracle.base import CitationOracleAdapter, OracleRouter
        from crucible.oracle.kernel_oracle import KernelOracle
        from crucible.orchestrator import Orchestrator
    except Exception:
        return None
    try:
        router = OracleRouter(default=KernelOracle()).register("existence_claim", CitationOracleAdapter())
        ledger = Ledger(os.environ.get("VERITAS_LEDGER_DB", str(REPO_ROOT / "veritas_ledger.db")))
        # Share OUR mission_id so the orchestrator's domain trace stitches to the
        # generator's proposal trace (both carry crucible.mission_id).
        orch = Orchestrator(
            oracle=router, ledger=ledger, mission_id=claim.mission_id,
            user_id="veritas-generator", convo_id="autoresearch-hackathon",
        )
        return orch.evaluate(claim, candidate, mission_name="kernel_generator")
    except Exception:
        print("[generator] orchestrator.evaluate raised (non-fatal, falling back):\n"
              + traceback.format_exc(), file=sys.stderr)
        return None


def _try_kernel_oracle(claim: Claim, candidate: Candidate) -> Optional[Verdict]:
    """FALLBACK (orchestrator absent): call modal-oracle's kernel oracle directly.

    ``KernelOracle().verify(claim, candidate)`` is the SINGLE canonical bridge — it
    takes pydantic Claim/Candidate and returns a pydantic ``schemas.Verdict``
    (oracle internals land in ``Verdict.evidence``), so no coercion is needed."""
    try:
        from crucible.oracle.kernel_oracle import KernelOracle
    except Exception:
        return None
    try:
        verdict = KernelOracle().verify(claim, candidate)
    except Exception:
        print("[generator] KernelOracle.verify raised (non-fatal):\n" + traceback.format_exc(),
              file=sys.stderr)
        return None
    return verdict if isinstance(verdict, Verdict) else None


def route_through_gate(claim: Claim, candidate: Candidate):
    """Run the candidate through the SAME gate as a rehearsed candidate. Returns
    ``(verdict, gate, path, outcome)`` — ``outcome`` is the orchestrator's
    ClaimOutcome when routed through it, else None. NO trust shortcut on any path."""
    # --- PREFERRED: crucible-core's full Orchestrator (oracle router + Modal +
    #     detector-D readback + truth-floor gate + ledger + certificate). ---
    outcome = _run_orchestrator(claim, candidate)
    if outcome is not None:
        return outcome.verdict, outcome.gate, "orchestrator", outcome

    # --- FALLBACK A: static pre-gate (local, before any GPU spend) ---
    from crucible.oracle.static_checker import static_pregate
    static = static_pregate(candidate.code or "", backend="triton", precision="fp32")
    if not static["ok"]:
        verdict = Verdict(
            claim_id=claim.claim_id, candidate_id=candidate.candidate_id, mission_id=claim.mission_id,
            verdict="blocked", oracle_type="kernel", verifier_status="OK",
            correctness_passed=False, tamper_detected=True, static_check_passed=False,
            blocked_reason=f"static pre-gate blocked: {static['blocked_reason']}",
            evidence={"static_pregate": static},
        )
        return verdict, evaluate_truth_floor(claim, verdict, trace_readback_confirmed=False), "static-blocked", None

    # --- FALLBACK B: call the kernel oracle directly (orchestrator unreachable) ---
    kv = _try_kernel_oracle(claim, candidate)
    if kv is not None:
        # Honest: without the orchestrator we did NOT perform the canonical
        # detector-D Workshop readback, so trace_readback_confirmed stays False
        # (even a confirmed oracle verdict will not 'commit' on this degraded path).
        gate = evaluate_truth_floor(claim, kv, trace_readback_confirmed=False)
        return kv, gate, "kernel-oracle(no-readback)", None

    # --- FALLBACK C: nothing reachable -> honest UNVERIFIED -> gate BLOCKS ---
    verdict = Verdict(
        claim_id=claim.claim_id, candidate_id=candidate.candidate_id, mission_id=claim.mission_id,
        verdict="unverified", oracle_type="kernel", verifier_status="OK",
        correctness_passed=False, tamper_detected=False, static_check_passed=True,
        speedup=None, speedup_threshold=claim.effective_threshold,
        blocked_reason="passed static pre-gate; no execution oracle reachable — candidate NOT trusted",
        evidence={"static_pregate": static, "note": "no oracle/orchestrator importable"},
    )
    return verdict, evaluate_truth_floor(claim, verdict, trace_readback_confirmed=False), "pending-modal-oracle", None


# --------------------------------------------------------------------------- #
# crucible.* domain spans for the generated-candidate flow (best-effort)
# --------------------------------------------------------------------------- #
def _emit_spans(claim: Claim, candidate: Candidate, verdict: Verdict, gate: GateResult) -> Optional[str]:
    try:
        from crucible.trace import CrucibleTracer
    except Exception:
        return None
    try:
        tracer = CrucibleTracer(
            mission_id=claim.mission_id,
            event_name="veritas_generator",
            user_id=os.environ.get("RAINDROP_USER_ID", "veritas-generator"),
            convo_id=os.environ.get("RAINDROP_CONVO_ID", "autoresearch-hackathon"),
        )
        mission = tracer.span("mission", "agent_root", "kernel_generator_mission")
        cand = tracer.span(
            "candidate", "llm_call", f"propose:{candidate.label}",
            claim_id=claim.claim_id, candidate_id=candidate.candidate_id,
            parent=mission, model=candidate.generator,
            input=claim.statement, output=candidate.strategy or "",
        )
        cand.finish("OK")
        verify = tracer.span(
            "verify", "tool_call", "truth_floor",
            claim_id=claim.claim_id, candidate_id=candidate.candidate_id,
            oracle_type=verdict.oracle_type if verdict.oracle_type in {
                "correctness", "speed", "anti_tamper", "replay", "citation"} else None,
            parent=mission,
        )
        verify.finish(
            "OK",
            verdict=verdict.verdict,
            correctness_passed=verdict.correctness_passed,
            tamper_detected=verdict.tamper_detected,
            blocked_reason=verdict.blocked_reason,
            promotion=gate.promotion,
        )
        mission.finish("OK")
        tracer.flush()
        return tracer.run_url
    except Exception as e:
        print(f"[generator] span emit failed (non-fatal): {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# 3. End-to-end: propose -> persist -> gate -> trace
# --------------------------------------------------------------------------- #
async def propose_and_gate(
    claim: Optional[Claim] = None,
    *,
    model: str = DEFAULT_MODEL,
    persist: bool = True,
    emit_spans: bool = True,
) -> GenerationResult:
    if claim is None:
        claim = Claim(
            mission_id=new_id("mis"),
            statement="A Triton RMSNorm kernel (reduce over dim=1) faster than the PyTorch reference.",
            claim_type="speedup_claim",
            target="36_RMSNorm",
            speedup_threshold=1.0,
        )

    from crucible.raindrop_bridge import install_raindrop_bridge

    bridge = install_raindrop_bridge(user_id="veritas-generator", convo_id="autoresearch-hackathon")
    try:
        candidate, proposal = await propose_candidate(claim, model=model)
    finally:
        if bridge is not None:
            bridge.flush()

    source_path = persist_candidate(candidate) if persist else None
    verdict, gate, gate_path, outcome = route_through_gate(claim, candidate)

    # The orchestrator emits its OWN crucible.* domain trace (claim/candidate/
    # verify/oracle/ledger), sharing our mission_id — so we don't double-emit.
    # Only emit our own spans on a fallback path where no orchestrator trace exists.
    if outcome is not None:
        try:
            from crucible.trace import BASE
            workshop_url = f"{BASE}/runs/{outcome.trace_id}"
        except Exception:
            workshop_url = None
    else:
        workshop_url = _emit_spans(claim, candidate, verdict, gate) if emit_spans else None

    return GenerationResult(
        candidate=candidate, proposal=proposal, verdict=verdict, gate=gate,
        source_path=source_path, gate_path=gate_path, workshop_url=workshop_url, outcome=outcome,
    )


def _print_report(res: GenerationResult) -> None:
    p = res.proposal
    print("\n  VERITAS — GENERATOR: a confident candidate, gated without mercy\n")
    print(f"  model           : {res.candidate.generator}")
    print(f"  strategy        : {p.strategy}")
    print(f"  claimed_speedup : {p.claimed_speedup}x  (the agent's confident claim)")
    print(f"  code            : {len(res.candidate.code or '')} chars, entry_point={res.candidate.entry_point}")
    print(f"  artifact_hash   : {res.candidate.artifact_hash}")
    if res.source_path:
        print(f"  persisted       : {res.source_path}")
    print(f"\n  GATE (same truth floor as rehearsed candidates — NO shortcut):")
    print(f"    path          : {res.gate_path}")
    print(f"    verdict       : {res.verdict.verdict}  (static_check_passed={res.verdict.static_check_passed})")
    print(f"    promotion     : {res.gate.promotion}   promoted={res.gate.promoted}")
    for r in res.gate.reasons:
        print(f"      - {r}")
    o = res.outcome
    if o is not None:
        if getattr(o, "speedup", None) is not None:
            print(f"    speedup       : {o.speedup}x  (MEASURED by a separate Modal oracle, not claimed)")
        if getattr(o, "promoted", False):
            ph = (getattr(o, "proof_hash", "") or "")[:16]
            print(f"    LEDGER COMMIT : ledger_id={o.ledger_id}  proof_hash={ph}…")
            if getattr(o, "certificate_id", None):
                print(f"    certificate   : {o.certificate_id}")
        if getattr(o, "trace_id", None):
            print(f"    trace_id      : {o.trace_id}")
    if res.verdict.blocked_reason:
        print(f"    blocked_reason: {res.verdict.blocked_reason}")
    if res.workshop_url:
        print(f"\n  Raindrop courtroom trace: {res.workshop_url}")

    if res.gate_path == "orchestrator" and res.gate.promoted:
        print("\n  >> CONFIRMED by a SEPARATE Modal oracle and COMMITTED to the verified ledger.")
        print("     Not trusted because the agent said so — reproduced under stated bounds.\n")
    elif res.gate_path == "orchestrator":
        print("\n  >> BLOCKED on merit by the same truth floor every candidate faces.")
        print(f"     {res.verdict.blocked_reason or 'see gate reasons above'}\n")
    elif res.gate_path == "pending-modal-oracle":
        print("\n  >> Generated candidate NOT trusted: passed static pre-gate, but no execution")
        print("     oracle was reachable, so the gate BLOCKS it. No self-certification, ever.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="VERITAS candidate-proposing generator")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--no-spans", action="store_true")
    args = ap.parse_args()

    # load .env (OPENAI_API_KEY + RAINDROP_WRITE_KEY)
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not set — cannot run the live generator.", file=sys.stderr)
        return 2

    try:
        res = asyncio.run(propose_and_gate(
            model=args.model, persist=not args.no_persist, emit_spans=not args.no_spans,
        ))
    except Exception:
        print("generator FAILED:\n" + traceback.format_exc(), file=sys.stderr)
        return 1

    _print_report(res)
    # Exit 0: the generator ran and the gate adjudicated. A blocked generated
    # candidate is a SUCCESSFUL demonstration of the no-trust-shortcut property.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
