"""crucible/oracle/base.py — the Oracle protocol (the seam crucible-core ⇄ modal-oracle).

The orchestrator dispatches every (Claim, Candidate) to an ``Oracle`` and gets a
:class:`crucible.schemas.Verdict` back.  modal-oracle's ``kernel_oracle`` and the
``citation_oracle`` both satisfy this; the :class:`OracleRouter` picks one per
claim_type.  This module is pure contract — no GPU, no network — so it imports
cleanly everywhere.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from crucible.schemas import Candidate, Claim, OracleType, Verdict


class OracleError(RuntimeError):
    """Raised when no oracle can handle a claim (routing failure)."""


@runtime_checkable
class Oracle(Protocol):
    """Anything that can turn a (claim, candidate) into a Verdict.

    Implementations MUST NOT raise for a *bad candidate* — that is a normal
    ``refuted`` verdict.  They should only raise for a *broken verifier*; the
    orchestrator catches that and records an ``ERROR`` verdict via
    :func:`make_error_verdict` so the truth floor blocks it (never silently
    swallowed).
    """

    name: str

    def verify(self, claim: Claim, candidate: Candidate) -> Verdict: ...


def make_error_verdict(
    claim: Claim,
    candidate: Candidate,
    exc: BaseException | str,
    *,
    oracle_type: OracleType = "kernel",
) -> Verdict:
    """Build the ``unverified`` / ``verifier_status=ERROR`` verdict for a verifier
    crash.  The error is surfaced verbatim — never hidden — and the gate blocks."""
    detail = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
    return Verdict(
        claim_id=claim.claim_id,
        candidate_id=candidate.candidate_id,
        mission_id=claim.mission_id,
        verdict="unverified",
        oracle_type=oracle_type,
        verifier_status="ERROR",
        correctness_passed=False,
        tamper_detected=False,
        blocked_reason="verifier error (see error field)",
        error=detail,
    )


class OracleRouter:
    """Route a claim to an oracle by ``claim_type`` (falling back to a default).

    Usage::

        router = OracleRouter(default=kernel_oracle)
        router.register("existence_claim", CitationOracleAdapter())
        verdict = router.verify(claim, candidate)
    """

    name = "router"

    def __init__(self, default: Optional[Oracle] = None, by_type: Optional[dict[str, Oracle]] = None):
        self.default = default
        self.by_type: dict[str, Oracle] = dict(by_type or {})

    def register(self, claim_type: str, oracle: Oracle) -> "OracleRouter":
        self.by_type[claim_type] = oracle
        return self

    def oracle_for(self, claim: Claim) -> Oracle:
        oracle = self.by_type.get(claim.claim_type, self.default)
        if oracle is None:
            raise OracleError(
                f"no oracle registered for claim_type {claim.claim_type!r} and no default"
            )
        return oracle

    def verify(self, claim: Claim, candidate: Candidate) -> Verdict:
        return self.oracle_for(claim).verify(claim, candidate)


class CitationOracleAdapter:
    """Adapt the existing ``citation_oracle.CitationOracle`` (which exposes
    ``check(citation) -> CitationCheck``) to the uniform ``Oracle`` protocol.

    Imports ``CitationOracle`` lazily so this module stays dependency-free until
    a citation oracle is actually used (the cold open)."""

    name = "citation"

    def __init__(self, citation_oracle=None, **kwargs):
        if citation_oracle is None:
            from crucible.oracle.citation_oracle import CitationOracle
            citation_oracle = CitationOracle(**kwargs)
        self._oracle = citation_oracle

    def verify(self, claim: Claim, candidate: Candidate) -> Verdict:
        check = self._oracle.check(claim.statement)
        verdict = check.to_verdict(
            claim_id=claim.claim_id,
            candidate_id=candidate.candidate_id,
            mission_id=claim.mission_id,
        )
        # to_verdict returns a Verdict when schemas are importable (always true here).
        if isinstance(verdict, Verdict):
            return verdict
        return Verdict(**verdict)  # defensive: standalone-dict fallback


__all__ = ["Oracle", "OracleError", "OracleRouter", "CitationOracleAdapter", "make_error_verdict"]
