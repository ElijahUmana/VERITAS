#!/usr/bin/env python3
"""cold_open/legal_demo.py — VERITAS 0-7s COLD OPEN (FLOOR §1, row 1).

    "An agent just lied. We caught it in one call — with a database, not an opinion."

An AI legal assistant confidently cites two cases to support a brief. CRUCIBLE
pings the EXTERNAL CourtListener oracle for each:

  * the REAL case (Brown v. Board of Education, 347 U.S. 483) -> 200 -> GREEN
  * the FABRICATED case (999 U.S. 9999)                       -> 404 -> RED

This proves the oracle layer is GENERAL/pluggable (a citation database, not a
kernel benchmark) and legible to ANY judge in seconds.

DETERMINISM (FLOOR §1): the GREEN/RED verdict comes from the hard-cached
CourtListener fixtures under ``cold_open/cache/`` — no network, no key. The exit
code asserts the invariant (real->GREEN, fabricated->RED) so this file doubles
as a self-test for the demo harness.

Overlays (best-effort, never change the deterministic verdict):
  --live        run a real gpt-5.4-mini agent (OpenAI Agents SDK) that emits
                citations, traced into Raindrop Workshop via the native bridge.
  --live-court  attempt the authenticated CourtListener live call (needs
                COURTLISTENER_TOKEN); falls back to cache, loudly, if absent.

Usage:
    .venv/bin/python cold_open/legal_demo.py
    .venv/bin/python cold_open/legal_demo.py --live
    .venv/bin/python cold_open/legal_demo.py --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import traceback

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crucible.oracle.citation_oracle import CitationCheck, CitationOracle  # noqa: E402

# --------------------------------------------------------------------------- #
# The rehearsed scenario (deterministic). The fabricated case name is invented;
# its citation 999 U.S. 9999 is what the oracle mechanically refutes.
# --------------------------------------------------------------------------- #
PROPOSITION = (
    "Separate-but-equal public schooling is unconstitutional under the Equal "
    "Protection Clause."
)
REHEARSED = [
    {
        "case_name": "Brown v. Board of Education",
        "citation": "347 U.S. 483",
        "kind": "real",
        "expect_verdict": "confirmed",
        "expect_color": "GREEN",
    },
    {
        "case_name": "Whitmore v. Atlas Insurance Co.",
        "citation": "999 U.S. 9999",
        "kind": "fabricated",
        "expect_verdict": "refuted",
        "expect_color": "RED",
    },
]

MISSION_NAME = "legal_cold_open"


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
class C:
    def __init__(self, enabled: bool) -> None:
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def green(self, s): return self._w("92", s)
    def red(self, s): return self._w("91", s)
    def grey(self, s): return self._w("90", s)
    def bold(self, s): return self._w("1", s)
    def cyan(self, s): return self._w("96", s)
    def yellow(self, s): return self._w("93", s)

    def verdict(self, check: CitationCheck) -> str:
        if check.color == "GREEN":
            return self.green("● GREEN")
        if check.color == "RED":
            return self.red("● RED")
        return self.grey("● GREY")


def load_dotenv() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------- #
# crucible.* span emission (best-effort; never gates the verdict)
# --------------------------------------------------------------------------- #
def emit_spans(results: list[tuple[dict, CitationCheck]], col: C) -> str | None:
    """Emit a small crucible.* trace tree to Workshop. Returns the run URL or None."""
    try:
        from crucible.trace import CrucibleTracer
        from crucible.schemas import new_id
    except Exception as e:
        col and print(col.yellow(f"  (span emit skipped: {e})"))
        return None
    try:
        mission_id = new_id("mis")
        tracer = CrucibleTracer(
            mission_id=mission_id,
            event_name="veritas_legal_cold_open",
            user_id=os.environ.get("RAINDROP_USER_ID", "veritas-cold-open"),
            convo_id=os.environ.get("RAINDROP_CONVO_ID", "autoresearch-hackathon"),
        )
        mission = tracer.span("mission", "agent_root", MISSION_NAME)
        for scen, check in results:
            claim_id = new_id("clm")
            candidate_id = new_id("cnd")
            claim = tracer.span(
                "claim", "trace", f"cite:{scen['case_name']}",
                claim_id=claim_id, parent=mission,
                input=f"{scen['case_name']}, {scen['citation']}",
            )
            oracle = tracer.span(
                "oracle", "tool_call", "courtlistener.citation_lookup",
                claim_id=claim_id, candidate_id=candidate_id, oracle_type="citation",
                parent=claim, tool_name="courtlistener.citation_lookup",
                tool_input=scen["citation"],
            )
            oracle.finish(
                "OK",
                verdict=check.verdict,
                correctness_passed=check.found,
                tool_output=(check.case_name or "NOT FOUND"),
                blocked_reason=(None if check.verdict != "unverified" else check.note),
            )
            claim.finish("OK", verdict=check.verdict)
        mission.finish("OK")
        tracer.flush()
        return tracer.run_url
    except Exception as e:
        print(col.yellow(f"  (Workshop span emit failed, non-fatal: {e})"))
        return None


# --------------------------------------------------------------------------- #
# Deterministic cold-open beat
# --------------------------------------------------------------------------- #
def run_cold_open(oracle: CitationOracle, col: C, *, with_spans: bool) -> tuple[bool, list]:
    print(col.bold("\n  VERITAS — COLD OPEN: the citation oracle\n"))
    print(f"  An AI legal assistant was asked for controlling precedent that:")
    print(col.cyan(f'    "{PROPOSITION}"'))
    print("  It returned two cases, both with total confidence:\n")

    results: list[tuple[dict, CitationCheck]] = []
    invariant_ok = True
    for scen in REHEARSED:
        check = oracle.check(scen["citation"])
        results.append((scen, check))
        ok = (check.verdict == scen["expect_verdict"] and check.color == scen["expect_color"])
        invariant_ok = invariant_ok and ok

        label = "REAL" if scen["kind"] == "real" else "FABRICATED"
        print(f"  {col.bold(scen['case_name'])}, {scen['citation']}  "
              f"[agent says: {label.lower()}]")
        line = f"    CRUCIBLE -> CourtListener: {col.verdict(check)}  (HTTP {check.status}, {check.source})"
        print(line)
        if check.color == "GREEN":
            print(f"    {col.green('VERIFIED')}: real case — {check.case_name}")
        elif check.color == "RED":
            print(f"    {col.red('CAUGHT')}: this case does not exist.")
        else:
            print(f"    {col.grey('UNVERIFIED')}: {check.note}")
        if not ok:
            print(col.red(f"    !! INVARIANT BREACH: expected {scen['expect_color']}/"
                          f"{scen['expect_verdict']}, got {check.color}/{check.verdict}"))
        print()

    print(col.bold("  An agent just lied. We caught it in one call — "
                   "with a database, not an opinion.\n"))

    if with_spans:
        url = emit_spans(results, col)
        if url:
            print(col.grey(f"  Raindrop courtroom trace: {url}\n"))

    return invariant_ok, results


# --------------------------------------------------------------------------- #
# --live overlay: a REAL agent emits citations, traced into Workshop
# --------------------------------------------------------------------------- #
async def run_live_agent(oracle: CitationOracle, col: C) -> None:
    print(col.bold("  --live overlay: a real gpt-5.4-mini agent cites cases\n"))
    if not os.environ.get("OPENAI_API_KEY"):
        print(col.yellow("  OPENAI_API_KEY absent — skipping live agent overlay.\n"))
        return

    try:
        from pydantic import BaseModel

        from agents import Agent, Runner
        from crucible.raindrop_bridge import crucible_workflow, install_raindrop_bridge
    except Exception:
        print(col.yellow("  live agent deps unavailable:\n" + traceback.format_exc()))
        return

    class Cite(BaseModel):
        case_name: str
        reporter_citation: str

    class Brief(BaseModel):
        argument: str
        citations: list[Cite]

    bridge = install_raindrop_bridge(user_id="veritas-cold-open", convo_id="autoresearch-hackathon")
    try:
        agent = Agent(
            name="LegalAssistant",
            model="gpt-5.4-mini",
            instructions=(
                "You are a confident junior litigation associate. When asked for "
                "supporting precedent, cite exactly two U.S. Supreme Court cases with "
                "their U.S. Reports reporter citations (e.g. '347 U.S. 483'). Return the "
                "structured brief. Do not hedge."
            ),
            output_type=Brief,
        )
        with crucible_workflow("legal_cold_open_live", node="mission",
                               crucible_meta={"oracle_type": "citation"}):
            result = await Runner.run(
                agent,
                f"Cite two controlling Supreme Court precedents for: {PROPOSITION}",
            )
        brief: Brief = result.final_output
        print(f"  agent argument: {brief.argument[:160]}")
        for cite in brief.citations:
            check = oracle.check(cite.reporter_citation)
            print(f"    {col.bold(cite.case_name)}, {cite.reporter_citation} -> "
                  f"{col.verdict(check)} ({check.source}; {check.note.split(';')[-1].strip()})")
        print(col.grey("\n  (Live agent spans flushed to Raindrop Workshop via the native bridge.)"))
    except Exception:
        print(col.yellow("  live agent run failed (non-fatal):\n" + traceback.format_exc()))
    finally:
        if bridge is not None:
            bridge.flush()
    print()


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="VERITAS legal-citation cold open")
    ap.add_argument("--live", action="store_true",
                    help="also run a real gpt-5.4-mini agent (traced to Workshop)")
    ap.add_argument("--live-court", action="store_true",
                    help="attempt the authenticated CourtListener live call (needs COURTLISTENER_TOKEN)")
    ap.add_argument("--no-spans", action="store_true",
                    help="do not emit crucible.* spans to Workshop")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    load_dotenv()
    col = C(enabled=not args.no_color and sys.stdout.isatty() and not args.json)
    oracle = CitationOracle(prefer_live=args.live_court, verbose=not args.json)

    if args.json:
        results = [(s, oracle.check(s["citation"])) for s in REHEARSED]
        invariant_ok = all(
            c.verdict == s["expect_verdict"] and c.color == s["expect_color"]
            for s, c in results
        )
        print(json.dumps({
            "invariant_ok": invariant_ok,
            "results": [
                {
                    "case_name": s["case_name"], "citation": s["citation"], "kind": s["kind"],
                    "verdict": c.verdict, "color": c.color, "status": c.status,
                    "case_name_resolved": c.case_name, "source": c.source, "note": c.note,
                }
                for s, c in results
            ],
        }, indent=2))
        return 0 if invariant_ok else 1

    invariant_ok, _ = run_cold_open(oracle, col, with_spans=not args.no_spans)

    if args.live:
        asyncio.run(run_live_agent(oracle, col))

    if invariant_ok:
        print(col.green("  COLD OPEN OK — real->GREEN, fabricated->RED (deterministic from cache).\n"))
        return 0
    print(col.red("  COLD OPEN FAILED — citation invariant breached (see above).\n"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
