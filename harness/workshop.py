"""Zero-dependency HTTP client + readback assertions for the local Raindrop Workshop.

This is the "courtroom verifier" used by both the demo runner and the adversarial
self-test. It deliberately uses ONLY the standard library (urllib) so it can never
be broken by a dependency issue and matches the proven phase-zero pattern
(phase-zero/integration/triad_smoke.py).

Verified live against Workshop build at :5899 (2026-05-30):
  • POST /api/traces/query   {sql,limit,max_bytes} -> {columns,rows,row_count,...}
                             json_extract on dotted attr keys works; attributes is JSON text.
  • GET  /api/runs                                  -> [ {id,event_id,event_name,...}, ... ]
  • GET  /api/annotations?run_id=<id>               -> [ {id,run_id,span_id,kind,note,source,created_at}, ... ]
  • POST /api/annotations    {run_id,kind,note,source,span_id?} -> 201 {id,...}
                             kind in {good,issue,note}; source in {claude-code,codex,user}.
  • DELETE /api/annotations/<id>                    -> {ok:true}
  • GET  /health                                    -> {ok:true,service:"workshop",...}

Run this module directly for a live end-to-end self-check of the client:
    .venv/bin/python harness/workshop.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Span attribute keys (the crucible.* span contract, FLOOR.md §2.1).
# Centralized so a rename by raindrop-courtroom is a one-line change here.
# ---------------------------------------------------------------------------
ATTR_MISSION = "crucible.mission_id"
ATTR_CLAIM = "crucible.claim_id"
ATTR_CANDIDATE = "crucible.candidate_id"
ATTR_NODE = "crucible.node"
ATTR_ORACLE_TYPE = "crucible.oracle_type"
ATTR_VERDICT = "crucible.verdict"
ATTR_PROMOTION = "crucible.promotion"
ATTR_TAMPER = "crucible.tamper_detected"
ATTR_BLOCKED_REASON = "crucible.blocked_reason"
ATTR_KIND = "raindrop.span.kind"

VALID_SOURCES = ("claude-code", "codex", "user")
VALID_KINDS = ("good", "issue", "note")


def _origin_from_env() -> str:
    """Workshop origin. Accepts RAINDROP_LOCAL_DEBUGGER like http://localhost:5899/v1/."""
    raw = os.environ.get("RAINDROP_LOCAL_DEBUGGER") or "http://localhost:5899"
    raw = raw.strip().rstrip("/")
    for suffix in ("/v1/traces", "/v1"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    return raw.rstrip("/") or "http://localhost:5899"


def jx(key: str, column: str = "attributes") -> str:
    """SQLite json_extract expression for a (possibly dotted) attribute key."""
    return f"json_extract({column}, '$.\"{key}\"')"


def sql_str(value: str) -> str:
    """Single-quote escape a string for inline SQL."""
    return "'" + str(value).replace("'", "''") + "'"


class WorkshopError(RuntimeError):
    pass


@dataclass
class WorkshopClient:
    origin: str = field(default_factory=_origin_from_env)
    timeout: float = 5.0
    # Endpoint fallbacks for forward/backward compat across Workshop builds.
    query_paths: tuple[str, ...] = ("/api/traces/query", "/query")

    # -- low-level -----------------------------------------------------------
    def _request(self, method: str, path: str, body: Any | None = None,
                 timeout: float | None = None) -> tuple[int, str]:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(self.origin + path, data=data,
                                     headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as exc:
            payload = ""
            try:
                payload = exc.read().decode()
            except Exception:
                pass
            return exc.code, payload

    def _json(self, method: str, path: str, body: Any | None = None) -> Any:
        code, text = self._request(method, path, body)
        if code >= 400:
            raise WorkshopError(f"{method} {path} -> HTTP {code}: {text[:240]}")
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise WorkshopError(f"{method} {path} returned non-JSON: {text[:160]}") from exc

    # -- health --------------------------------------------------------------
    def health(self) -> dict:
        return self._json("GET", "/health")

    def is_up(self) -> bool:
        try:
            return bool(self.health().get("ok"))
        except Exception:
            return False

    # -- query ---------------------------------------------------------------
    def query(self, sql: str, *, limit: int = 200, max_bytes: int = 400_000) -> list[dict]:
        """Run a read-only SELECT against the Workshop trace DB. Returns rows."""
        body = {"sql": sql, "limit": limit, "max_bytes": max_bytes}
        last_err = None
        for path in self.query_paths:
            code, text = self._request("POST", path, body)
            if code in (404, 405):
                last_err = f"{path}: HTTP {code}"
                continue
            if code >= 400:
                raise WorkshopError(f"query {path} -> HTTP {code}: {text[:240]}")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise WorkshopError(f"query {path} non-JSON: {text[:160]}") from exc
            rows = data.get("rows", []) if isinstance(data, dict) else []
            return rows if isinstance(rows, list) else []
        raise WorkshopError(f"no working query endpoint ({last_err})")

    def query_one(self, sql: str) -> dict | None:
        rows = self.query(sql, limit=1)
        return rows[0] if rows else None

    def scalar(self, sql: str) -> Any:
        row = self.query_one(sql)
        if not row:
            return None
        return next(iter(row.values()), None)

    # -- runs ----------------------------------------------------------------
    def runs(self, limit: int = 50) -> list[dict]:
        data = self._json("GET", "/api/runs")
        rows = data if isinstance(data, list) else data.get("rows", []) if isinstance(data, dict) else []
        return rows[:limit]

    # -- annotations ---------------------------------------------------------
    def get_annotations(self, run_id: str) -> list[dict]:
        q = urllib.parse.urlencode({"run_id": run_id})
        data = self._json("GET", f"/api/annotations?{q}")
        return data if isinstance(data, list) else []

    def write_annotation(self, run_id: str, kind: str, note: str,
                         source: str = "claude-code", span_id: str | None = None) -> str:
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")
        body = {"run_id": run_id, "kind": kind, "note": note, "source": source}
        if span_id:
            body["span_id"] = span_id
        data = self._json("POST", "/api/annotations", body)
        return data.get("id") if isinstance(data, dict) else None

    def delete_annotation(self, annotation_id: str) -> bool:
        code, _ = self._request("DELETE", f"/api/annotations/{annotation_id}")
        return code in (200, 204)

    # -- run discovery -------------------------------------------------------
    def find_run_by_mission(self, mission_id: str, *, since_ms: int | None = None) -> str | None:
        """Find the run whose spans carry crucible.mission_id == mission_id."""
        where = [f"{jx(ATTR_MISSION)} = {sql_str(mission_id)}"]
        if since_ms is not None:
            where.append(f"r.last_updated_at >= {int(since_ms)}")
        sql = (
            "SELECT DISTINCT s.run_id AS run_id, r.last_updated_at AS upd "
            "FROM spans s JOIN runs r ON r.id = s.run_id "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY r.last_updated_at DESC"
        )
        row = self.query_one(sql)
        return row.get("run_id") if row else None

    def find_run_by_event(self, event_name: str, *, since_ms: int | None = None) -> str | None:
        where = [f"event_name = {sql_str(event_name)}"]
        if since_ms is not None:
            where.append(f"last_updated_at >= {int(since_ms)}")
        sql = (f"SELECT id FROM runs WHERE {' AND '.join(where)} "
               "ORDER BY last_updated_at DESC")
        row = self.query_one(sql)
        return row.get("id") if row else None

    # -- claim-level readback (the core verification primitives) ------------
    def claim_span_ids(self, run_id: str, claim_id: str) -> list[str]:
        sql = (f"SELECT id FROM spans WHERE run_id = {sql_str(run_id)} "
               f"AND {jx(ATTR_CLAIM)} = {sql_str(claim_id)}")
        return [r["id"] for r in self.query(sql) if r.get("id")]

    def claim_oracle_spans(self, run_id: str, claim_id: str) -> list[dict]:
        """Spans for this claim that represent an oracle verification step."""
        sql = (
            "SELECT id, name, status, "
            f"{jx(ATTR_ORACLE_TYPE)} AS oracle_type, {jx(ATTR_NODE)} AS node, "
            f"{jx(ATTR_VERDICT)} AS verdict "
            f"FROM spans WHERE run_id = {sql_str(run_id)} "
            f"AND {jx(ATTR_CLAIM)} = {sql_str(claim_id)} "
            f"AND ({jx(ATTR_NODE)} = 'oracle' OR {jx(ATTR_ORACLE_TYPE)} IS NOT NULL)"
        )
        return self.query(sql)

    def issue_annotations_for_claim(self, run_id: str, claim_id: str) -> list[dict]:
        """Issue annotations whose span belongs to this claim (join span_id->crucible.claim_id)."""
        span_ids = set(self.claim_span_ids(run_id, claim_id))
        out = []
        for a in self.get_annotations(run_id):
            if a.get("kind") != "issue":
                continue
            sid = a.get("span_id")
            # A claim-scoped issue is one attached to one of the claim's spans.
            # Run-level issues (span_id is None) are treated as run-scoped, not claim-scoped.
            if sid and sid in span_ids:
                out.append(a)
        return out

    def assert_promoted_clean(self, run_id: str, claim_id: str) -> tuple[bool, dict]:
        """A PROMOTED claim must have >=1 oracle span AND zero issue annotations."""
        oracle = self.claim_oracle_spans(run_id, claim_id)
        issues = self.issue_annotations_for_claim(run_id, claim_id)
        detail = {
            "claim_id": claim_id,
            "oracle_span_count": len(oracle),
            "oracle_spans": [{"name": o.get("name"), "oracle_type": o.get("oracle_type"),
                              "verdict": o.get("verdict")} for o in oracle],
            "issue_annotation_count": len(issues),
        }
        ok = len(oracle) >= 1 and len(issues) == 0
        return ok, detail

    def assert_rejected_flagged(self, run_id: str, claim_id: str) -> tuple[bool, dict]:
        """A REJECTED claim must have >=1 issue annotation attached to one of its spans."""
        issues = self.issue_annotations_for_claim(run_id, claim_id)
        detail = {
            "claim_id": claim_id,
            "issue_annotation_count": len(issues),
            "issue_notes": [a.get("note", "")[:120] for a in issues],
        }
        return len(issues) >= 1, detail

    # -- polling helper ------------------------------------------------------
    def wait_for(self, predicate, *, timeout: float = 12.0, interval: float = 0.5):
        """Poll predicate() until truthy or timeout. Returns last value."""
        deadline = time.monotonic() + timeout
        value = None
        while True:
            try:
                value = predicate()
            except Exception:
                value = None
            if value:
                return value
            if time.monotonic() >= deadline:
                return value
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Live self-check
# ---------------------------------------------------------------------------
def _selfcheck() -> int:
    c = WorkshopClient()
    print(f"workshop origin: {c.origin}")
    ok_all = True

    def chk(label, cond, extra=""):
        nonlocal ok_all
        mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
        ok_all = ok_all and bool(cond)
        print(f"  {mark}  {label}{(' — ' + extra) if extra else ''}")

    h = {}
    try:
        h = c.health()
    except Exception as exc:
        chk("health", False, str(exc))
    chk("health ok", bool(h.get("ok")), json.dumps(h))

    rows = c.query("SELECT COUNT(*) AS n FROM runs")
    n_runs = rows[0]["n"] if rows else None
    chk("query runs count", isinstance(n_runs, int), f"{n_runs} runs")

    rows = c.query("SELECT COUNT(*) AS n FROM spans WHERE "
                   f"{jx(ATTR_KIND)} IS NOT NULL")
    chk("json_extract on dotted key", bool(rows), f"spans w/ raindrop.span.kind={rows[0]['n'] if rows else '?'}")

    runs = c.runs(limit=3)
    chk("GET /api/runs", isinstance(runs, list) and len(runs) >= 1, f"{len(runs)} returned")

    # annotation round-trip on the most recent run (write -> read -> delete)
    target = runs[0]["id"] if runs else None
    if target:
        try:
            aid = c.write_annotation(target, "note",
                                     "harness.workshop self-check (auto-deleted)",
                                     source="claude-code")
            chk("write_annotation", bool(aid), f"id={aid}")
            found = any(a.get("id") == aid for a in c.get_annotations(target))
            chk("read back annotation", found)
            deleted = c.delete_annotation(aid) if aid else False
            chk("delete_annotation", deleted)
            gone = not any(a.get("id") == aid for a in c.get_annotations(target))
            chk("annotation removed", gone)
        except Exception as exc:
            chk("annotation round-trip", False, str(exc))

    print("\033[32mworkshop client OK\033[0m" if ok_all else "\033[31mworkshop client FAILED\033[0m")
    return 0 if ok_all else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
