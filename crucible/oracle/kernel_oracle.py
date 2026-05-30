"""VERITAS kernel oracle — client-side wrapper over the deployed Modal verifier.

VERITAS-original. This is the crucible-core-facing API: it implements the Oracle protocol
`verify(claim, candidate) -> Verdict(dict)` for the KernelBench RMSNorm lab.

Flow (FLOOR.md §2.2/§2.3):
  1. STATIC PRE-GATE client-side (`static_checker.static_pregate`). If it blocks, return a
     `blocked` verdict WITHOUT calling Modal — disguise cheats cost zero GPU seconds.
  2. Otherwise look up the deploy-once Modal function (`modal.Function.from_name`) and dispatch
     the candidate source + job spec. The verifier measures everything itself and returns a
     JSON-only verdict; we never trust candidate-reported numbers.

The returned dict is the Verdict contract shared with crucible-core (proposed via SendMessage):
  verdict ∈ {confirmed, refuted, blocked, unverified}; correctness_passed; speedup;
  tamper_detected; verifier_status ∈ {OK, ERROR}; blocked_reason; hardware; measured_by;
  details; error.  No Modal/torch import is required just to run the static gate.
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Optional

from . import static_checker

APP_NAME = "veritas-verifier"
FUNCTION_NAME = "verify_candidate"

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LAB = _ROOT / "benchmarks" / "rmsnorm_lab"


def reference_source() -> str:
    """The reference source string fed to the oracle (honors VERITAS_RMSNORM_SHAPE)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("veritas_reference", _LAB / "reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reference_source()


def candidate_source(name: str) -> str:
    """Load a candidate by file stem from benchmarks/rmsnorm_lab/candidates/."""
    path = _LAB / "candidates" / (name if name.endswith(".py") else f"{name}.py")
    return path.read_text()


def get_verifier():
    """Look up the deploy-once Modal verifier function. Clear error if not yet deployed."""
    import modal

    try:
        return modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    except Exception as e:  # noqa: BLE001 — surface a precise, actionable message
        raise RuntimeError(
            f"Modal function {APP_NAME}/{FUNCTION_NAME} not found ({e}). "
            f"Deploy once with:  modal deploy modal/verifier_app.py"
        ) from e


def _blocked_verdict(claim_id, candidate_id, static: dict) -> dict:
    return {
        "verdict": "blocked", "claim_id": claim_id, "candidate_id": candidate_id,
        "oracle_type": "kernel", "correctness_passed": False, "speedup": None,
        "tamper_detected": True, "verifier_status": "OK",
        "blocked_reason": f"static pre-gate: {static['blocked_reason']}",
        "hardware": None, "measured_by": "modal-oracle",
        "details": {"static": static}, "error": None,
    }


def verify(
    claim: dict,
    candidate: dict,
    *,
    spec: Optional[dict] = None,
    skip_static: bool = False,
) -> dict:
    """Verify one candidate. Returns the Verdict contract (dict).

    claim:     {claim_id, claim_type ("speedup_claim"), ...}
    candidate: {candidate_id, source (str), backend ("triton"), [reference_src]}
    spec:      optional oracle overrides (tolerance, num_perf_trials, thresholds, ...)
    """
    claim_id = str(claim.get("claim_id", "claim"))
    candidate_id = str(candidate.get("candidate_id", candidate.get("name", "candidate")))
    backend = str(candidate.get("backend", "triton"))
    precision = str((spec or {}).get("precision", "fp32"))
    src = candidate.get("source")
    if src is None and candidate.get("name"):
        src = candidate_source(candidate["name"])
    if not src:
        raise ValueError("candidate must include 'source' (str) or 'name' (file stem)")

    # 1) STATIC PRE-GATE (client-side) — disguise/bypass cheats die before any GPU spend.
    static = static_checker.static_pregate(src, backend=backend, precision=precision)
    if not skip_static and not static["ok"]:
        return _blocked_verdict(claim_id, candidate_id, static)

    # 2) Dispatch to the deploy-once Modal verifier.
    payload = {
        "reference_src": candidate.get("reference_src") or reference_source(),
        "candidate_src": src,
        "claim_id": claim_id,
        "candidate_id": candidate_id,
        "backend": backend,
        "precision": precision,
    }
    if spec:
        for k in ("tolerance", "num_correct_trials", "num_warmup", "num_perf_trials",
                  "seed", "dual_timer_threshold", "excessive_speedup_threshold", "run_static"):
            if k in spec:
                payload[k] = spec[k]

    verifier = get_verifier()
    result = verifier.remote(payload)
    # Defensive: ensure the contract keys exist even if the worker returned something odd.
    result.setdefault("claim_id", claim_id)
    result.setdefault("candidate_id", candidate_id)
    result.setdefault("measured_by", "modal-oracle")
    return result


def to_verdict(result: dict, claim: Any, candidate: Any):
    """Adapt the Modal worker's verdict dict into a `crucible.schemas.Verdict` (pydantic).

    Lazy import so this module stays usable without pydantic/schemas (e.g. the dict-only proof
    path). Only the contract's defined fields are set (Verdict is extra="forbid"); the full
    worker `details` go into the `evidence` escape hatch.
    """
    from crucible.schemas import AntiTamperResult, CorrectnessResult, SpeedResult, Verdict

    details = result.get("details") or {}
    static = details.get("static") or {}
    speed = details.get("speed") or {}
    at = details.get("anti_tamper") or {}
    corr = details.get("correctness") or {}

    is_speed = bool(getattr(claim, "is_speedup_claim", False))
    threshold = getattr(claim, "effective_threshold", None) if is_speed else None

    speed_model = None
    if speed:
        speed_model = SpeedResult(
            speedup=float(result.get("speedup") or speed.get("speedup_do_bench") or 0.0),
            ref_time_ms=speed.get("ref_do_bench_ms"),
            candidate_time_ms=speed.get("cand_do_bench_ms"),
            warmup=int(speed.get("num_warmup", 0) or 0),
            trials=int(speed.get("num_perf_trials", 0) or 0),
            threshold=threshold,
        )
    at_model = None
    if at:
        reasons = [r for r in [result.get("blocked_reason")] if r]
        at_model = AntiTamperResult(
            tamper_detected=bool(result.get("tamper_detected", False)),
            static_check_passed=static.get("ok"),
            dual_timer_ratio=at.get("dual_timer_ratio"),
            excessive_speedup=bool(at.get("excessive_speedup", False)),
            reasons=reasons,
        )
    corr_model = None
    if corr:
        trials = list(corr.values())
        errs = [t.get("max_abs_err") for t in trials if t.get("max_abs_err") is not None]
        corr_model = CorrectnessResult(
            passed=bool(result.get("correctness_passed", False)),
            trials=len(trials),
            atol=1e-2, rtol=1e-2,
            max_abs_err=(max(errs) if errs else None),
            mismatches=[t.get("reason") for t in trials if not t.get("passed") and t.get("reason")],
        )

    hw = result.get("hardware")
    return Verdict(
        claim_id=getattr(claim, "claim_id", result.get("claim_id", "claim")),
        candidate_id=getattr(candidate, "candidate_id", result.get("candidate_id", "candidate")),
        mission_id=getattr(claim, "mission_id", "mission"),
        verdict=result.get("verdict", "unverified"),
        oracle_type="kernel",
        verifier_status=result.get("verifier_status", "OK"),
        correctness_passed=bool(result.get("correctness_passed", False)),
        tamper_detected=bool(result.get("tamper_detected", False)),
        speedup=result.get("speedup"),
        speedup_threshold=threshold,
        static_check_passed=static.get("ok"),
        blocked_reason=result.get("blocked_reason"),
        error=result.get("error"),
        hardware=(f"Modal {hw}" if hw else None),
        correctness=corr_model,
        speed=speed_model,
        anti_tamper=at_model,
        evidence={"details": details, "measured_by": result.get("measured_by", "modal-oracle")},
    )


class KernelOracle:
    """Oracle-protocol object (see crucible/oracle/base.py): `verify(Claim, Candidate) -> Verdict`.

    Accepts pydantic Claim/Candidate (the orchestrator seam) OR plain dicts (convenience).
    Per the protocol contract it does NOT raise for a bad candidate (that's a `refuted`/`blocked`
    verdict); only a broken verifier surfaces as verdict=unverified / verifier_status=ERROR.
    """

    name = "kernel_oracle"
    oracle_type = "kernel"

    def __init__(self, spec: Optional[dict] = None):
        self.spec = spec or {}

    def verify(self, claim: Any, candidate: Any):
        # Pull primitives from pydantic models or dicts.
        src = getattr(candidate, "code", None) if not isinstance(candidate, dict) else candidate.get("source")
        source_path = getattr(candidate, "source_path", None) if not isinstance(candidate, dict) else candidate.get("source_path")
        if not src and source_path:
            src = pathlib.Path(source_path).read_text()
        if not src and (getattr(candidate, "label", None) or (isinstance(candidate, dict) and candidate.get("name"))):
            src = candidate_source(getattr(candidate, "label", None) or candidate["name"])

        meta = getattr(candidate, "metadata", None) or (candidate.get("metadata") if isinstance(candidate, dict) else {}) or {}
        cand_dict = {
            "candidate_id": getattr(candidate, "candidate_id", None) or (candidate.get("candidate_id") if isinstance(candidate, dict) else None),
            "source": src,
            "backend": meta.get("backend", "triton"),
        }
        claim_dict = {
            "claim_id": getattr(claim, "claim_id", None) or (claim.get("claim_id") if isinstance(claim, dict) else None),
            "claim_type": getattr(claim, "claim_type", None) or (claim.get("claim_type") if isinstance(claim, dict) else "speedup_claim"),
        }
        result = verify(claim_dict, cand_dict, spec=self.spec)

        # Typed return when given pydantic models (the orchestrator path); dict otherwise.
        if isinstance(claim, dict) and isinstance(candidate, dict):
            return result
        return to_verdict(result, claim, candidate)


if __name__ == "__main__":
    import json

    name = os.environ.get("VERITAS_CANDIDATE", "good_rehearsed")
    out = verify({"claim_id": "cli", "claim_type": "speedup_claim"},
                 {"candidate_id": name, "name": name})
    print(json.dumps(out, indent=2))
