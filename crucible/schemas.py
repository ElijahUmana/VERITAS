"""VERITAS / CRUCIBLE — the shared data contract.

Every floor builder plugs into these models:
  * ``modal-oracle``     produces a :class:`Verdict` from a (Claim, Candidate).
  * ``openai-generator`` produces a :class:`Candidate` (and often the :class:`Claim`).
  * ``raindrop-courtroom`` reads the ``crucible.*`` span attributes (see ``trace.py``)
    and the :class:`Verdict` fields the detectors gate on.
  * ``crucible-core``    (this package) gates promotion (:func:`evaluate_truth_floor`),
    writes the :class:`Certificate`, and appends the :class:`LedgerRow`.

Design rules (FLOOR.md §2):
  * Gate-critical models (``Verdict``) use ``extra="forbid"`` so a field-name drift
    between teams fails LOUD instead of silently mis-gating. Use the ``evidence`` /
    ``metadata`` dicts as the explicit extensibility escape hatches.
  * Bounded language only — a confirmed claim is "verified under stated bounds",
    never "proved correct" (see ``certificate.py``).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Canonical enumerations (FLOOR.md §2.1).  Exposed as constants AND used as
# ``Literal`` types so consumers get both clean JSON and import-time symbols.
# --------------------------------------------------------------------------- #
CLAIM_TYPES = ("speedup_claim", "correctness_claim", "existence_claim", "general")
ClaimType = Literal["speedup_claim", "correctness_claim", "existence_claim", "general"]

# The verdict label that lands on ``crucible.verdict``.
VERDICTS = ("confirmed", "refuted", "blocked", "unverified")
VerdictLabel = Literal["confirmed", "refuted", "blocked", "unverified"]

# Oracle families.  FLOOR §2.1 enumerates the per-span oracle types; "kernel"
# is the composite the Modal execution oracle returns (correctness+speed+tamper),
# and test/proof are reserved for the CEILING oracle layer.
ORACLE_TYPES = ("correctness", "speed", "anti_tamper", "replay", "citation", "kernel", "test", "proof")
OracleType = Literal["correctness", "speed", "anti_tamper", "replay", "citation", "kernel", "test", "proof"]

VERIFIER_STATUSES = ("OK", "ERROR", "TIMEOUT")
VerifierStatus = Literal["OK", "ERROR", "TIMEOUT"]

# What happened at the promotion gate -> ``crucible.promotion``.
PROMOTIONS = ("committed", "blocked", "replayed", "regressed")
Promotion = Literal["committed", "blocked", "replayed", "regressed"]


# --------------------------------------------------------------------------- #
# Small helpers (shared id / time / hash conventions).
# --------------------------------------------------------------------------- #
def new_id(prefix: str) -> str:
    """Stable, sortable-ish unique id with a human-readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    """UTC ISO-8601 timestamp (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_text(text: str) -> str:
    """Hex sha256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(obj: Any) -> str:
    """Deterministic sha256 over a JSON-serialisable object (sorted keys)."""
    return sha256_text(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False))


class _Base(BaseModel):
    """Base config shared by the structured sub-results."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Structured evidence sub-results (FLOOR.md §2.2).  All optional so the oracle
# can fill them incrementally; the flat ``Verdict`` fields remain authoritative
# for the gate.
# --------------------------------------------------------------------------- #
class Assumptions(_Base):
    """The stated bounds a verdict holds under (FLOOR §2.4 certificate)."""

    shape: Optional[str] = None          # e.g. "(8192, 8192) fp32"
    dtype: Optional[str] = None          # e.g. "torch.float32"
    hardware: Optional[str] = None       # e.g. "Modal Tesla T4"
    tolerance: Optional[str] = None      # e.g. "fp32 atol=rtol=1e-2"
    seeds: list[int] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class OracleProtocol(_Base):
    """Human-readable description of the protocol actually applied (FLOOR §2.2)."""

    correctness: Optional[str] = None    # "5 trials, seeds from 42, torch.allclose atol=rtol=1e-2, pristine recompute"
    speed: Optional[str] = None          # "5 warmup, 100 trials, cuda_event timing, L2 clears, speedup=ref/cand"
    anti_tamper: Optional[str] = None    # "dual-timer >1.5x reject, >10x excessive reject, static pre-gate"


class CorrectnessResult(_Base):
    """Result of the correctness oracle (FLOOR §2.2)."""

    passed: bool
    trials: int = 0
    seeds: list[int] = Field(default_factory=list)
    atol: Optional[float] = None
    rtol: Optional[float] = None
    max_abs_err: Optional[float] = None
    mismatches: list[str] = Field(default_factory=list)   # per-trial failure descriptions


class SpeedResult(_Base):
    """Result of the speed oracle (FLOOR §2.2)."""

    speedup: float                        # ref_time / candidate_time
    ref_time_ms: Optional[float] = None
    candidate_time_ms: Optional[float] = None
    warmup: int = 0
    trials: int = 0
    threshold: Optional[float] = None     # required-to-beat ratio (echo of the claim)


class AntiTamperResult(_Base):
    """Result of the anti-tamper courtroom checks (FLOOR §2.2)."""

    tamper_detected: bool
    static_check_passed: Optional[bool] = None
    dual_timer_ratio: Optional[float] = None   # cuda_event vs do_bench disagreement; >1.5x => reject
    excessive_speedup: bool = False            # >10x => flag + reject
    reasons: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Claim — what an agent asserts.
# --------------------------------------------------------------------------- #
class Claim(_Base):
    claim_id: str = Field(default_factory=lambda: new_id("clm"))
    mission_id: str
    statement: str                                   # human-readable, e.g. "A 2x faster RMSNorm kernel"
    claim_type: ClaimType = "general"
    target: str                                      # what is being improved, e.g. "36_RMSNorm"
    speedup_threshold: Optional[float] = None        # ratio a speedup_claim must reach (gate uses >=; default 1.0)
    baseline_ledger_id: Optional[str] = None         # compounding: prior verified row this builds on
    assumptions: Assumptions = Field(default_factory=Assumptions)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)

    @property
    def is_speedup_claim(self) -> bool:
        return self.claim_type == "speedup_claim"

    @property
    def effective_threshold(self) -> float:
        """The speedup ratio the gate requires (defaults to 1.0 = 'not slower')."""
        return self.speedup_threshold if self.speedup_threshold is not None else 1.0


# --------------------------------------------------------------------------- #
# Candidate — a concrete artifact proposed to satisfy a claim.
# --------------------------------------------------------------------------- #
class Candidate(_Base):
    candidate_id: str = Field(default_factory=lambda: new_id("cnd"))
    claim_id: str
    mission_id: str
    source_path: Optional[str] = None        # path to the candidate module/file on disk
    entry_point: Optional[str] = None        # symbol the oracle calls (e.g. "ModelNew" / function name)
    code: Optional[str] = None               # inline source (the generator emits this)
    artifact_hash: Optional[str] = None      # sha256 of code/artifact; auto-filled from ``code`` if absent
    generator: str = "unknown"               # "gpt-5.4-mini" | "rehearsed" | ...
    strategy: Optional[str] = None           # short description of the approach
    label: Optional[str] = None              # e.g. "good_rehearsed", "wrong_lastdim", "tamper_result_reuse"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)

    @model_validator(mode="after")
    def _fill_hash(self) -> "Candidate":
        if self.code and not self.artifact_hash:
            # bypass validate_assignment recursion via object.__setattr__-safe path
            self.artifact_hash = sha256_text(self.code)
        return self


# --------------------------------------------------------------------------- #
# Verdict — the oracle's output.  GATE-CRITICAL.  Co-designed with modal-oracle.
# Flat fields (correctness_passed / tamper_detected / speedup / verifier_status)
# are authoritative for the truth floor; structured sub-results carry evidence.
# --------------------------------------------------------------------------- #
class Verdict(_Base):
    claim_id: str
    candidate_id: str
    mission_id: str

    verdict: VerdictLabel                          # the headline -> crucible.verdict
    oracle_type: OracleType = "kernel"             # which oracle family produced this
    verifier_status: VerifierStatus = "OK"         # "ERROR" means the verifier itself failed

    # --- flat gate inputs (authoritative for evaluate_truth_floor) ---
    correctness_passed: bool = False
    tamper_detected: bool = False
    speedup: Optional[float] = None                # candidate/baseline ratio (None for non-speed claims)
    speedup_threshold: Optional[float] = None      # echo of the claim's required ratio
    static_check_passed: Optional[bool] = None

    blocked_reason: Optional[str] = None           # short reason -> crucible.blocked_reason
    error: Optional[str] = None                    # surfaced verifier error (never swallowed)
    hardware: Optional[str] = None                 # e.g. "Modal Tesla T4"

    # --- structured evidence (optional; fill what applies) ---
    correctness: Optional[CorrectnessResult] = None
    speed: Optional[SpeedResult] = None
    anti_tamper: Optional[AntiTamperResult] = None
    oracle_protocol: Optional[OracleProtocol] = None
    assumptions: Optional[Assumptions] = None
    evidence: dict[str, Any] = Field(default_factory=dict)   # free-form escape hatch
    created_at: str = Field(default_factory=now_iso)

    @property
    def is_error(self) -> bool:
        return self.verifier_status == "ERROR"


# --------------------------------------------------------------------------- #
# The promotion gate result (FLOOR §2.3 "truth floor").
# --------------------------------------------------------------------------- #
class GateResult(_Base):
    promoted: bool
    promotion: Promotion
    reasons: list[str] = Field(default_factory=list)            # per-condition explanation
    failed_conditions: list[str] = Field(default_factory=list)  # condition keys that failed
    blocked_reason: Optional[str] = None


def evaluate_truth_floor(
    claim: Claim,
    verdict: Verdict,
    trace_readback_confirmed: bool,
) -> GateResult:
    """The single source of truth for FLOOR.md §2.3.

    A claim enters the ledger as ``committed`` ONLY if ALL hold::

        correctness_passed == true
        tamper_detected == false
        oracle_verdict == confirmed
        verifier_status != ERROR
        (speedup >= threshold) OR claim_type != speedup_claim
        trace_readback_confirmed == true        # Raindrop detector D: oracle span exists

    Otherwise the result is ``blocked`` and the candidate is retained as negative
    evidence.  This function is pure (no IO) and is the canonical encoding of the
    spec — ``orchestrator.py`` calls it; tests assert against it.
    """
    reasons: list[str] = []
    failed: list[str] = []

    def check(key: str, ok: bool, ok_msg: str, fail_msg: str) -> None:
        if ok:
            reasons.append(f"PASS {key}: {ok_msg}")
        else:
            reasons.append(f"FAIL {key}: {fail_msg}")
            failed.append(key)

    check(
        "correctness_passed",
        verdict.correctness_passed is True,
        "candidate matched the reference oracle",
        "candidate did not match the reference oracle",
    )
    check(
        "no_tamper",
        verdict.tamper_detected is False,
        "no tamper detected",
        f"tamper detected ({verdict.blocked_reason or 'anti-tamper triggered'})",
    )
    check(
        "oracle_confirmed",
        verdict.verdict == "confirmed",
        "oracle verdict is confirmed",
        f"oracle verdict is '{verdict.verdict}'",
    )
    check(
        "verifier_ok",
        verdict.verifier_status != "ERROR",
        f"verifier status {verdict.verifier_status}",
        f"verifier errored ({verdict.error or 'no detail'})",
    )

    if claim.is_speedup_claim:
        threshold = claim.effective_threshold
        has_speed = verdict.speedup is not None
        meets = has_speed and verdict.speedup >= threshold
        check(
            "speedup_meets_threshold",
            meets,
            f"speedup {verdict.speedup}x >= {threshold}x",
            (
                f"speedup {verdict.speedup}x < {threshold}x"
                if has_speed
                else f"no speedup measured for a speedup_claim (threshold {threshold}x)"
            ),
        )
    else:
        reasons.append(f"PASS speedup_meets_threshold: not a speedup_claim ({claim.claim_type})")

    check(
        "trace_readback_confirmed",
        trace_readback_confirmed is True,
        "claim has an oracle span in Workshop (detector D)",
        "no oracle span found in Workshop for this claim (detector D)",
    )

    promoted = not failed
    if promoted:
        return GateResult(
            promoted=True,
            promotion="committed",
            reasons=reasons,
            failed_conditions=[],
            blocked_reason=None,
        )

    blocked_reason = verdict.blocked_reason or f"truth floor failed: {', '.join(failed)}"
    return GateResult(
        promoted=False,
        promotion="blocked",
        reasons=reasons,
        failed_conditions=failed,
        blocked_reason=blocked_reason,
    )


# --------------------------------------------------------------------------- #
# Certificate — the judge-facing artifact (FLOOR §2.4).  Bounded language.
# --------------------------------------------------------------------------- #
BOUNDED_LANGUAGE = (
    "Verified under the stated bounds and accepted under this oracle. "
    "This is not a claim of universal correctness."
)


class Certificate(_Base):
    certificate_id: str = Field(default_factory=lambda: new_id("crt"))
    claim_id: str
    candidate_id: str
    mission_id: str
    claim: str                                   # the statement
    claim_type: ClaimType
    artifact_hash: str
    assumptions: Assumptions
    oracle_protocol: OracleProtocol
    verdict: VerdictLabel
    speedup: Optional[float] = None
    trace_id: str
    run_id: Optional[int] = None
    ledger_id: Optional[str] = None
    proof_hash: str
    statement_of_bounds: str = BOUNDED_LANGUAGE
    created_at: str = Field(default_factory=now_iso)


# --------------------------------------------------------------------------- #
# LedgerRow — a row in the SQLite verified ledger (FLOOR §2).  Stores BOTH
# committed increments (compounding baselines) and blocked candidates
# (negative evidence so run #2 skips refuted paths).
# --------------------------------------------------------------------------- #
class LedgerRow(_Base):
    ledger_id: str = Field(default_factory=lambda: new_id("ldg"))
    mission_id: str
    claim_id: str
    candidate_id: str
    run_id: int                                  # compounding run number (1, 2, ...)
    claim: str
    claim_type: ClaimType
    target: str                                  # benchmark/target key for run#2 retrieval
    artifact_hash: str
    artifact_path: Optional[str] = None
    verdict: VerdictLabel
    promotion: Promotion                         # committed | blocked | replayed | regressed
    speedup: Optional[float] = None
    baseline_speedup: Optional[float] = None     # what it improved over (compounding)
    parent_ledger_id: Optional[str] = None       # the prior verified row it built on
    proof_hash: str
    trace_id: str
    certificate_id: Optional[str] = None
    blocked_reason: Optional[str] = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    committed_at: str = Field(default_factory=now_iso)

    @property
    def is_committed(self) -> bool:
        return self.promotion in ("committed", "replayed")


__all__ = [
    # enums / constants
    "CLAIM_TYPES", "ClaimType", "VERDICTS", "VerdictLabel",
    "ORACLE_TYPES", "OracleType", "VERIFIER_STATUSES", "VerifierStatus",
    "PROMOTIONS", "Promotion", "BOUNDED_LANGUAGE",
    # helpers
    "new_id", "now_iso", "sha256_text", "canonical_hash",
    # sub-results
    "Assumptions", "OracleProtocol", "CorrectnessResult", "SpeedResult", "AntiTamperResult",
    # core models
    "Claim", "Candidate", "Verdict", "GateResult", "Certificate", "LedgerRow",
    # the gate
    "evaluate_truth_floor",
]
