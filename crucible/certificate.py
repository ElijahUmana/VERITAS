"""crucible/certificate.py — the VERITAS Claim Certificate (FLOOR.md §2.4).

The judge-facing artifact for a verdict.  Emitted as JSON (machine) + Markdown
(human).  Language is deliberately BOUNDED: a confirmed claim is "verified under
the stated bounds" / "accepted under this oracle" — NEVER "proved correct".
The ``proof_hash`` binds the certificate to the exact artifact + bounds + verdict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from crucible.schemas import (
    BOUNDED_LANGUAGE,
    Assumptions,
    Candidate,
    Certificate,
    Claim,
    OracleProtocol,
    Verdict,
    canonical_hash,
)

# Default protocol descriptions (FLOOR §2.2) used when the oracle doesn't supply
# its own — so a certificate always states HOW the verdict was reached.
_DEFAULT_PROTOCOL = OracleProtocol(
    correctness=(
        "reference forward vs candidate over 5 trials (seeds from 42) + hidden extra "
        "shape/seed; torch.allclose at fp32 atol=rtol=1e-2; candidate runs on cloned "
        "inputs, reference recomputed from the pristine copy; shape/dtype/isnan/isinf asserted."
    ),
    speed=(
        "cuda.synchronize -> 5 warmup -> 100 timed trials with L2 clears between trials; "
        "speedup = reference_time / candidate_time on the same harness."
    ),
    anti_tamper=(
        "dual timer (cuda_event vs do_bench) rejects >1.5x disagreement (stream bypass); "
        ">10x speedup rejected (timing fraud); static pre-gate blocks torch-in-disguise / "
        "try-except / bare-pass before any GPU spend."
    ),
)


def _merge_assumptions(claim: Claim, verdict: Verdict) -> Assumptions:
    """Prefer the oracle's measured assumptions; fall back to the claim's; fill
    hardware from the verdict if known."""
    base = verdict.assumptions or claim.assumptions or Assumptions()
    a = base.model_copy(deep=True)
    if not a.hardware and verdict.hardware:
        a.hardware = verdict.hardware
    return a


def build_certificate(
    claim: Claim,
    candidate: Candidate,
    verdict: Verdict,
    *,
    trace_id: str,
    run_id: Optional[int] = None,
    ledger_id: Optional[str] = None,
) -> Certificate:
    """Assemble a Certificate and compute its ``proof_hash``.

    The proof_hash is a deterministic digest over the verified essentials: the
    claim, the artifact hash, the verdict + speedup, the stated bounds, and the
    oracle protocol.  Two certificates with the same essentials hash identically.
    """
    assumptions = _merge_assumptions(claim, verdict)
    protocol = verdict.oracle_protocol or _DEFAULT_PROTOCOL
    artifact_hash = candidate.artifact_hash or ""

    proof_hash = canonical_hash({
        "claim": claim.statement,
        "claim_type": claim.claim_type,
        "artifact_hash": artifact_hash,
        "verdict": verdict.verdict,
        "speedup": verdict.speedup,
        "assumptions": assumptions.model_dump(),
        "oracle_protocol": protocol.model_dump(),
        "oracle_type": verdict.oracle_type,
    })

    return Certificate(
        claim_id=claim.claim_id,
        candidate_id=candidate.candidate_id,
        mission_id=claim.mission_id,
        claim=claim.statement,
        claim_type=claim.claim_type,
        artifact_hash=artifact_hash,
        assumptions=assumptions,
        oracle_protocol=protocol,
        verdict=verdict.verdict,
        speedup=verdict.speedup,
        trace_id=trace_id,
        run_id=run_id,
        ledger_id=ledger_id,
        proof_hash=proof_hash,
    )


def render_markdown(cert: Certificate) -> str:
    """Render the human-facing Markdown certificate with bounded language."""
    a = cert.assumptions
    p = cert.oracle_protocol
    speed_line = f"{cert.speedup:.3f}×" if cert.speedup is not None else "n/a (not a speedup claim)"
    headline = {
        "confirmed": "✅ VERIFIED UNDER STATED BOUNDS",
        "refuted": "❌ REFUTED BY THE ORACLE",
        "blocked": "⛔ BLOCKED BEFORE PROMOTION",
        "unverified": "⚠️ UNVERIFIED (no decisive oracle result)",
    }.get(cert.verdict, cert.verdict.upper())

    lines = [
        "# VERITAS Claim Certificate",
        "",
        f"**Status:** {headline}",
        "",
        f"- **Certificate ID:** `{cert.certificate_id}`",
        f"- **Claim ID:** `{cert.claim_id}`",
        f"- **Candidate ID:** `{cert.candidate_id}`",
        f"- **Mission ID:** `{cert.mission_id}`",
        f"- **Run #:** {cert.run_id if cert.run_id is not None else 'n/a'}",
        "",
        "## Claim",
        f"> {cert.claim}",
        "",
        f"- **Type:** `{cert.claim_type}`",
        f"- **Verdict:** `{cert.verdict}`",
        f"- **Measured speedup:** {speed_line}",
        "",
        "## Artifact",
        f"- **Artifact hash (sha256):** `{cert.artifact_hash}`",
        "",
        "## Assumptions (the stated bounds this verdict holds under)",
        f"- **Shape:** {a.shape or 'unspecified'}",
        f"- **Dtype:** {a.dtype or 'unspecified'}",
        f"- **Hardware:** {a.hardware or 'unspecified'}",
        f"- **Tolerance:** {a.tolerance or 'unspecified'}",
    ]
    if a.seeds:
        lines.append(f"- **Seeds:** {a.seeds}")
    lines += [
        "",
        "## Oracle protocol applied",
        f"- **Correctness:** {p.correctness or 'n/a'}",
        f"- **Speed:** {p.speed or 'n/a'}",
        f"- **Anti-tamper:** {p.anti_tamper or 'n/a'}",
        "",
        "## Provenance",
        f"- **Raindrop trace / run ID:** `{cert.trace_id}`",
        f"- **Ledger ID:** `{cert.ledger_id or 'not promoted'}`",
        f"- **Proof hash:** `{cert.proof_hash}`",
        f"- **Issued:** {cert.created_at}",
        "",
        "---",
        f"_{cert.statement_of_bounds}_",
    ]
    return "\n".join(lines)


def write_certificate(cert: Certificate, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``<certificate_id>.json`` and ``<certificate_id>.md`` to ``out_dir``.
    Returns ``(json_path, md_path)``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{cert.certificate_id}.json"
    md_path = out / f"{cert.certificate_id}.md"
    json_path.write_text(json.dumps(json.loads(cert.model_dump_json()), indent=2) + "\n")
    md_path.write_text(render_markdown(cert) + "\n")
    return json_path, md_path


__all__ = ["build_certificate", "render_markdown", "write_certificate"]
