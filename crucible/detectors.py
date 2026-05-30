#!/usr/bin/env python3
"""crucible/detectors.py — VERITAS / CRUCIBLE Raindrop courtroom detectors.

The 4 SQL detectors (FLOOR.md §2.3) that turn Raindrop Workshop into the
COURTROOM: every claim is tried against the trace record, and contract
violations are caught by SELECT, not by opinion. Each detector is a single
read-only SQLite SELECT run against the Workshop query API
(``POST /api/traces/query``) — the exact same surface as the raindrop MCP
``query_traces`` tool, so anything here can be re-verified by hand in Workshop.

    A — SILENT VERIFY CONTRADICTION
        node=verify ∧ verdict=confirmed ∧ status=ERROR
        (the verifier *claimed* confirmed but its own span errored → hallucinated
         verification / silent failure)

    B — UNSUPPORTED PROMOTION  (the auditor that proves the gate held)
        node=ledger ∧ promotion=committed ∧ verdict≠confirmed
        (something entered the ledger without a confirmed oracle verdict — this
         set MUST be empty if the promotion gate works)

    C — TAMPER
        oracle_type=anti_tamper ∧ tamper_detected=1
        (the anti-tamper oracle caught a reward-hack: result-reuse, zeroed
         inputs, stream bypass, torch-in-disguise, excessive speedup)

    D — NO-ORACLE
        {claims} EXCEPT {claims that have an oracle-bearing span}
        (a claim that reached a verdict without ANY external mechanical oracle —
         not independently verifiable → feeds §2.3 trace_readback_confirmed)

Run from the package root:
    python -m crucible.detectors <run_id>
or import:
    from crucible.detectors import run_all_detectors, adjudicate, annotate_from_report
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

try:  # share the daemon base URL with the emitter when imported as a package
    from crucible.trace import BASE
except Exception:  # standalone execution / different import path
    _raw = os.environ.get("RAINDROP_LOCAL_DEBUGGER", "http://localhost:5899/v1/").rstrip("/")
    BASE = _raw[:-3] if _raw.endswith("/v1") else _raw

__all__ = [
    "run_detector", "run_all_detectors", "adjudicate", "annotate",
    "annotate_from_report", "DETECTORS", "BASE",
]

_RUN_ID_RE = re.compile(r"^[0-9a-fA-F\-]{8,64}$")


# --- HTTP ------------------------------------------------------------------
def _post(path, obj, base=None):
    url = (base or BASE).rstrip("/") + path
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach Workshop daemon at {url}: {e}") from e


def _query(sql, base=None):
    """Run one read-only SELECT via the Workshop query API; return list[dict]."""
    st, body = _post("/api/traces/query", {"sql": sql}, base=base)
    if st != 200:
        raise RuntimeError(f"query failed ({st}): {body}\n--- SQL ---\n{sql}")
    data = json.loads(body) if body else {}
    return data.get("rows", data.get("data", []) or [])


def _run_clause(run_id, alias="s"):
    if not run_id:
        return ""
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"unsafe run_id {run_id!r}")
    return f"\n  AND {alias}.run_id='{run_id}'"


# --- the 4 canonical detector SQL templates (inspectable / copy-pasteable) --
# Each selects spans.id AS span_id so findings can be annotated on the exact
# offending span. ``{run}`` is replaced with an optional run scope.
DETECTOR_A_SILENT = """SELECT s.id AS span_id, s.run_id AS run_id,
  json_extract(s.attributes,'$."crucible.claim_id"') AS claim_id,
  json_extract(s.attributes,'$."crucible.verdict"')  AS verdict,
  s.name AS span_name, s.status AS status
FROM spans s
WHERE json_extract(s.attributes,'$."crucible.node"')='verify'
  AND json_extract(s.attributes,'$."crucible.verdict"')='confirmed'
  AND s.status='ERROR'{run}"""

DETECTOR_B_UNSUPPORTED = """SELECT s.id AS span_id, s.run_id AS run_id,
  json_extract(s.attributes,'$."crucible.claim_id"')  AS claim_id,
  json_extract(s.attributes,'$."crucible.promotion"') AS promotion,
  json_extract(s.attributes,'$."crucible.verdict"')   AS verdict,
  json_extract(s.attributes,'$."crucible.ledger_id"') AS ledger_id,
  s.name AS span_name
FROM spans s
WHERE json_extract(s.attributes,'$."crucible.node"')='ledger'
  AND json_extract(s.attributes,'$."crucible.promotion"')='committed'
  AND IFNULL(json_extract(s.attributes,'$."crucible.verdict"'),'') <> 'confirmed'{run}"""

DETECTOR_C_TAMPER = """SELECT s.id AS span_id, s.run_id AS run_id,
  json_extract(s.attributes,'$."crucible.claim_id"')        AS claim_id,
  json_extract(s.attributes,'$."crucible.candidate_id"')    AS candidate_id,
  json_extract(s.attributes,'$."crucible.tamper_detected"') AS tamper_detected,
  json_extract(s.attributes,'$."crucible.blocked_reason"')  AS blocked_reason,
  s.name AS span_name
FROM spans s
WHERE json_extract(s.attributes,'$."crucible.oracle_type"')='anti_tamper'
  AND json_extract(s.attributes,'$."crucible.tamper_detected"')=1{run}"""

# D: claims that exist but have NO oracle-bearing span. "Has an oracle span" is
# defined as "some span for this claim carries a crucible.oracle_type" — robust
# across correctness/speed/anti_tamper/citation/replay oracles.
DETECTOR_D_NO_ORACLE = """SELECT DISTINCT json_extract(s.attributes,'$."crucible.claim_id"') AS claim_id
FROM spans s
WHERE json_extract(s.attributes,'$."crucible.node"') IN ('claim','candidate')
  AND json_extract(s.attributes,'$."crucible.claim_id"') IS NOT NULL{run}
EXCEPT
SELECT DISTINCT json_extract(s.attributes,'$."crucible.claim_id"') AS claim_id
FROM spans s
WHERE json_extract(s.attributes,'$."crucible.oracle_type"') IS NOT NULL
  AND json_extract(s.attributes,'$."crucible.claim_id"') IS NOT NULL{run}"""

DETECTORS = {
    "A_silent": DETECTOR_A_SILENT,
    "B_unsupported": DETECTOR_B_UNSUPPORTED,
    "C_tamper": DETECTOR_C_TAMPER,
    "D_no_oracle": DETECTOR_D_NO_ORACLE,
}

_ROSTER = """SELECT s.id AS span_id,
  json_extract(s.attributes,'$."crucible.node"')            AS node,
  json_extract(s.attributes,'$."crucible.claim_id"')        AS claim_id,
  json_extract(s.attributes,'$."crucible.candidate_id"')    AS candidate_id,
  json_extract(s.attributes,'$."crucible.verdict"')         AS verdict,
  json_extract(s.attributes,'$."crucible.promotion"')       AS promotion,
  json_extract(s.attributes,'$."crucible.oracle_type"')     AS oracle_type,
  json_extract(s.attributes,'$."crucible.tamper_detected"') AS tamper_detected,
  s.status AS status, s.name AS span_name
FROM spans s
WHERE json_extract(s.attributes,'$."crucible.mission_id"') IS NOT NULL{run}
ORDER BY s.start_time_ms ASC"""


def sql_for(name, run_id=None):
    """Return the exact SQL string for a detector (handy for MCP query_traces)."""
    tmpl = DETECTORS[name]
    return tmpl.replace("{run}", _run_clause(run_id))


def run_detector(name, run_id=None, base=None):
    return _query(sql_for(name, run_id), base=base)


def run_all_detectors(run_id=None, base=None):
    return {name: run_detector(name, run_id, base=base) for name in DETECTORS}


# --- adjudication: turn detector rows + roster into per-claim verdicts ------
def adjudicate(run_id, base=None):
    """Run all 4 detectors + a span roster for ``run_id`` and produce a per-claim
    courtroom verdict plus the annotation directives to write.

    Returns a dict:
      detectors  : raw rows per detector
      claims     : {claim_id: {...verdict fields..., annotation}}
      directives : [{run_id, span_id, kind, note}] to feed annotate_from_report
      gate_held  : bool — detector B empty (no unsupported promotion got through)
      caught     : counts of A/C/D findings (the courtroom working as designed)
    """
    dets = run_all_detectors(run_id, base=base)
    roster = _query(_ROSTER.replace("{run}", _run_clause(run_id)), base=base)

    claims = {}

    def entry(cid):
        return claims.setdefault(cid, {
            "claim_id": cid, "verdict": None, "ledger_verdict": None, "promotion": None,
            "oracle_present": False, "tamper": False, "silent": False,
            "unsupported_promotion": False, "no_oracle": False, "issues": [],
            "claim_span": None, "oracle_span": None, "ledger_span": None,
            "antitamper_span": None, "verify_error_span": None,
        })

    for r in roster:
        cid = r.get("claim_id")
        if cid is None:
            continue
        e = entry(cid)
        node = r.get("node")
        if node == "claim":
            e["claim_span"] = r["span_id"]
        if node == "oracle":
            e["oracle_present"] = True
            e["oracle_span"] = e["oracle_span"] or r["span_id"]
            if r.get("verdict"):
                e["verdict"] = r["verdict"]
        if r.get("oracle_type"):  # any oracle-family span proves oracle backing
            e["oracle_present"] = True
            e["oracle_span"] = e["oracle_span"] or r["span_id"]
        if node == "anti_tamper":
            e["antitamper_span"] = r["span_id"]
        if node == "ledger":
            e["ledger_span"] = r["span_id"]
            e["promotion"] = r.get("promotion")
            e["ledger_verdict"] = r.get("verdict")
        if node == "verify" and r.get("status") == "ERROR":
            e["verify_error_span"] = r["span_id"]

    for r in dets["A_silent"]:
        e = entry(r.get("claim_id")); e["silent"] = True
        e["issues"].append("silent verifier contradiction (verdict=confirmed but verify span ERRORED)")
        e["verify_error_span"] = r.get("span_id") or e["verify_error_span"]
    for r in dets["C_tamper"]:
        e = entry(r.get("claim_id")); e["tamper"] = True
        e["issues"].append("anti-tamper oracle fired (tamper_detected=1) — reward-hack blocked")
        e["antitamper_span"] = r.get("span_id") or e["antitamper_span"]
    for r in dets["B_unsupported"]:
        e = entry(r.get("claim_id")); e["unsupported_promotion"] = True
        e["issues"].append("UNSUPPORTED PROMOTION — committed to ledger without a confirmed verdict")
        e["ledger_span"] = r.get("span_id") or e["ledger_span"]
    for r in dets["D_no_oracle"]:
        e = entry(r.get("claim_id")); e["no_oracle"] = True; e["oracle_present"] = False
        e["issues"].append("no external oracle span — claim not independently verifiable")

    directives = []
    for cid, e in claims.items():
        committed = (e["promotion"] == "committed")
        verified = (e["ledger_verdict"] == "confirmed") or (e["verdict"] == "confirmed")
        clean = not (e["tamper"] or e["silent"] or e["unsupported_promotion"]) and e["oracle_present"]
        e["promotable"] = committed and clean and verified
        if e["issues"]:
            span = (e["antitamper_span"] or e["verify_error_span"] or e["ledger_span"]
                    or e["oracle_span"] or e["claim_span"])
            note = f"{cid} BLOCKED — " + "; ".join(e["issues"])
            directives.append({"run_id": run_id, "span_id": span, "kind": "issue", "note": note})
            e["annotation"] = "issue"
        elif committed and verified and e["oracle_present"]:
            span = e["ledger_span"] or e["oracle_span"] or e["claim_span"]
            note = (f"{cid} VERIFIED — promoted with a confirmed oracle verdict, no tamper, "
                    f"no silent failure; a separate oracle reproduced the result.")
            directives.append({"run_id": run_id, "span_id": span, "kind": "good", "note": note})
            e["annotation"] = "good"
        else:
            e["annotation"] = None

    return {
        "run_id": run_id,
        "detectors": dets,
        "claims": claims,
        "directives": directives,
        "gate_held": len(dets["B_unsupported"]) == 0,
        "caught": {"A_silent": len(dets["A_silent"]), "C_tamper": len(dets["C_tamper"]),
                   "D_no_oracle": len(dets["D_no_oracle"])},
    }


# --- annotations (programmatic path; orchestrator uses these) ---------------
def annotate(run_id, kind, note, span_id=None, base=None, source="claude-code"):
    """Write a durable run/span annotation (issue|good|note) via POST /api/annotations.

    Same endpoint the raindrop MCP ``annotate`` tool writes to.
    """
    if kind not in ("issue", "good", "note"):
        raise ValueError(f"kind must be issue|good|note, got {kind!r}")
    body = {"run_id": run_id, "kind": kind, "note": note, "source": source}
    if span_id:
        body["span_id"] = span_id
    st, resp = _post("/api/annotations", body, base=base)
    if st not in (200, 201):
        raise RuntimeError(f"annotate failed ({st}): {resp}")
    try:
        return json.loads(resp)
    except Exception:
        return {"status": st, "raw": resp}


def annotate_from_report(report, base=None):
    """Write all good/issue annotations a report produced. Returns written rows."""
    out = []
    for d in report["directives"]:
        out.append(annotate(d["run_id"], d["kind"], d["note"], span_id=d.get("span_id"), base=base))
    return out


def judge_and_annotate(run_id, base=None, write=True):
    """One call for the orchestrator: run the 4 detectors, adjudicate per-claim,
    and (optionally) write the good/issue courtroom annotations. Returns the report
    so a fresh demo run is ALWAYS fully annotated (deterministic; survives Workshop
    `clear` because re-emitting + re-judging reproduces the whole audit trail).
    """
    rep = adjudicate(run_id, base=base)
    if write:
        rep["annotations_written"] = annotate_from_report(rep, base=base)
    return rep


def trace_readback_confirmed(run_id, claim_id, base=None):
    """FLOOR §2.3 promotion-gate input: True iff ``claim_id`` has at least one
    oracle span in the trace (i.e. NOT flagged by detector D). The orchestrator
    ANDs this into the truth-floor before committing to the ledger."""
    no_oracle = {r.get("claim_id") for r in run_detector("D_no_oracle", run_id, base=base)}
    return claim_id not in no_oracle


def _print_report(report):
    r = report
    print(f"\n=== CRUCIBLE COURTROOM — run {r['run_id']} ===")
    for name in DETECTORS:
        rows = r["detectors"][name]
        flag = "·" if not rows else "⚑"
        print(f"  [{flag}] detector {name:<14} {len(rows)} row(s)")
        for row in rows:
            print(f"        {row}")
    print(f"\n  gate_held (detector B empty) = {r['gate_held']}")
    print(f"  caught (courtroom working)   = {r['caught']}")
    print("\n  per-claim verdicts:")
    for cid, e in r["claims"].items():
        print(f"    {cid:<8} promotion={e['promotion']} verdict={e['verdict'] or e['ledger_verdict']} "
              f"oracle={e['oracle_present']} promotable={e['promotable']} annotation={e.get('annotation')}")
        for issue in e["issues"]:
            print(f"             ⚑ {issue}")
    print("\n  annotation directives:")
    for d in r["directives"]:
        print(f"    {d['kind']:<5} span={d['span_id']} :: {d['note']}")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else None
    do_write = "--annotate" in sys.argv[2:]
    if not run:
        print("usage: python -m crucible.detectors <run_id> [--annotate]")
        print("\nCanonical detector SQL (paste into raindrop MCP query_traces):\n")
        for name in DETECTORS:
            print(f"--- {name} ---\n{sql_for(name)}\n")
        sys.exit(0)
    rep = adjudicate(run)
    _print_report(rep)
    if do_write:
        written = annotate_from_report(rep)
        print(f"\n  wrote {len(written)} annotation(s).")
