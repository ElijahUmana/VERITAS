"""crucible/ledger.py — the SQLite verified compounding ledger (FLOOR.md §2).

The ledger is the layer no shipped autoresearch system has: **verified memory that
compounds**.  It stores:

  * ``committed`` rows  — increments that PASSED the truth floor (run N+1's baselines)
  * ``blocked`` rows    — candidates that FAILED, retained as NEGATIVE EVIDENCE so the
                          generator can avoid the pattern and run N+1 skips the refuted path

Read-back is lossless (the full :class:`~crucible.schemas.LedgerRow` JSON is stored in a
``payload`` column and reconstructed on read), which is how we prove "persistent state
actually persisting across runs" (FLOOR DONE checklist).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from crucible.schemas import LedgerRow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    ledger_id        TEXT PRIMARY KEY,
    mission_id       TEXT NOT NULL,
    claim_id         TEXT NOT NULL,
    candidate_id     TEXT NOT NULL,
    run_id           INTEGER NOT NULL,
    claim            TEXT NOT NULL,
    claim_type       TEXT NOT NULL,
    target           TEXT NOT NULL,
    artifact_hash    TEXT NOT NULL,
    artifact_path    TEXT,
    verdict          TEXT NOT NULL,
    promotion        TEXT NOT NULL,
    speedup          REAL,
    baseline_speedup REAL,
    parent_ledger_id TEXT,
    proof_hash       TEXT NOT NULL,
    trace_id         TEXT NOT NULL,
    certificate_id   TEXT,
    blocked_reason   TEXT,
    committed_at     TEXT NOT NULL,
    payload          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_target     ON claims(target);
CREATE INDEX IF NOT EXISTS idx_claims_promotion  ON claims(promotion);
CREATE INDEX IF NOT EXISTS idx_claims_claim_id   ON claims(claim_id);
CREATE INDEX IF NOT EXISTS idx_claims_run_id     ON claims(run_id);
"""

# A row counts as a usable verified baseline iff its promotion is one of these.
_COMMITTED = ("committed", "replayed")


class Ledger:
    """A SQLite-backed verified ledger.  Safe to open the same ``db_path`` across
    process runs — that is exactly how run #2 reads run #1's increments."""

    def __init__(self, db_path: str | Path = "veritas_ledger.db"):
        self.db_path = str(db_path)
        # autocommit (isolation_level=None) + WAL for concurrent read-back during a run.
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(_SCHEMA)

    # ---- write ------------------------------------------------------------ #
    def record(self, row: LedgerRow) -> LedgerRow:
        """Insert (or replace by ledger_id) a ledger row — committed OR blocked."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO claims (
                ledger_id, mission_id, claim_id, candidate_id, run_id, claim, claim_type,
                target, artifact_hash, artifact_path, verdict, promotion, speedup,
                baseline_speedup, parent_ledger_id, proof_hash, trace_id, certificate_id,
                blocked_reason, committed_at, payload
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row.ledger_id, row.mission_id, row.claim_id, row.candidate_id, row.run_id,
                row.claim, row.claim_type, row.target, row.artifact_hash, row.artifact_path,
                row.verdict, row.promotion, row.speedup, row.baseline_speedup,
                row.parent_ledger_id, row.proof_hash, row.trace_id, row.certificate_id,
                row.blocked_reason, row.committed_at, row.model_dump_json(),
            ),
        )
        return row

    # ---- read-back (lossless) -------------------------------------------- #
    @staticmethod
    def _row(r: sqlite3.Row) -> LedgerRow:
        return LedgerRow.model_validate_json(r["payload"])

    def get(self, ledger_id: str) -> Optional[LedgerRow]:
        r = self.conn.execute("SELECT payload FROM claims WHERE ledger_id=?", (ledger_id,)).fetchone()
        return self._row(r) if r else None

    # explicit read-back alias used by the acceptance/demo to prove persistence
    read_back = get

    def by_claim(self, claim_id: str) -> list[LedgerRow]:
        rows = self.conn.execute(
            "SELECT payload FROM claims WHERE claim_id=? ORDER BY committed_at", (claim_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def by_candidate(self, candidate_id: str) -> Optional[LedgerRow]:
        r = self.conn.execute(
            "SELECT payload FROM claims WHERE candidate_id=? ORDER BY committed_at DESC LIMIT 1",
            (candidate_id,),
        ).fetchone()
        return self._row(r) if r else None

    def all(self) -> list[LedgerRow]:
        rows = self.conn.execute("SELECT payload FROM claims ORDER BY run_id, committed_at").fetchall()
        return [self._row(r) for r in rows]

    # ---- compounding (run #2 reads run #1) ------------------------------- #
    def committed_for_target(self, target: str) -> list[LedgerRow]:
        """All verified increments for a target, best speedup first (then most recent)."""
        rows = self.conn.execute(
            f"""
            SELECT payload FROM claims
            WHERE target=? AND promotion IN ({','.join('?' * len(_COMMITTED))})
            ORDER BY COALESCE(speedup, 0) DESC, committed_at DESC
            """,
            (target, *_COMMITTED),
        ).fetchall()
        return [self._row(r) for r in rows]

    def latest_baseline(self, target: str) -> Optional[LedgerRow]:
        """The verified row run N+1 should build on: best confirmed speedup for the
        target (ties broken by recency).  ``None`` if nothing verified yet."""
        rows = self.committed_for_target(target)
        return rows[0] if rows else None

    def negative_evidence(self, target: str) -> list[LedgerRow]:
        """Blocked/refuted candidates for a target — the refuted paths run N+1 skips."""
        rows = self.conn.execute(
            "SELECT payload FROM claims WHERE target=? AND promotion='blocked' ORDER BY committed_at DESC",
            (target,),
        ).fetchall()
        return [self._row(r) for r in rows]

    def refuted_artifact_hashes(self, target: str) -> set[str]:
        """Artifact hashes already proven to fail for this target (dedup / skip)."""
        return {r.artifact_hash for r in self.negative_evidence(target)}

    def next_run_id(self) -> int:
        """The run number for a NEW run = max(existing)+1, else 1 (compounding clock)."""
        r = self.conn.execute("SELECT MAX(run_id) AS m FROM claims").fetchone()
        return int(r["m"]) + 1 if r and r["m"] is not None else 1

    # ---- counts / lifecycle ---------------------------------------------- #
    def counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT promotion, COUNT(*) AS n FROM claims GROUP BY promotion"
        ).fetchall()
        out = {r["promotion"]: int(r["n"]) for r in rows}
        out["total"] = sum(out.values())
        return out

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["Ledger"]
