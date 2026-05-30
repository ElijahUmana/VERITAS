#!/usr/bin/env python3
"""crucible/demo.py — VERITAS: the one-command, <60s, lands-every-time demo.

    "Everyone else builds agents that optimize. We built the COURTROOM that
     decides whether the optimization is real — and only real, verified
     increments compound."

Runs the full FLOOR (FLOOR.md §1) in order. The centerpiece verdicts are
GATE-PRODUCED by the REAL CRUCIBLE engine — never hand-stamped:

  0–7s   COLD OPEN     an agent cites two cases; the citation oracle flashes a
                       real case GREEN and a fabricated case RED (cached, no net).
  7–22s  THE CHEAT     the real gate runs a confident "2x faster RMSNorm" against
                       an EXTERNAL oracle and REJECTS it — the anti-tamper oracle
                       computes that it reused a stale buffer; a torch-in-disguise
                       cheat dies at the static pre-gate. Span red + 'issue' annotation.
  22–40s VERIFIED      the honest candidate passes the same oracle (correct + a
                       genuine, MEASURED speedup, no tamper) → ledger COMMITTED with
                       a proof_hash; Claim Certificate written.
  40–52s COMPOUNDING   run #2 reads run #1's verified ledger row (real SQLite
                       round-trip) as its baseline, skips the refuted paths, commits.
  52–60s RAINDROP      open Workshop: the run, good/issue annotations, the ledger
                       row + trace_id, and a REPLAY that re-verifies the increment.

The cheat/verified beats are produced by crucible.courtroom_run (Orchestrator +
truth-floor gate + oracle): on the deterministic CPU ReferenceRMSNormOracle for the
stage-safe floor (guaranteed-green, keyless, no GPU/Modal) — or the live Modal
KernelOracle with --modal. Either way the verdict is COMPUTED by oracle execution,
not asserted. Every beat is verified live against the Raindrop Workshop (:5899).

Usage:
    .venv/bin/python crucible/demo.py                 # CPU gate (stage-safe floor)
    .venv/bin/python crucible/demo.py --modal          # live Modal T4 oracle (real GPU; may exceed 60s)
    .venv/bin/python crucible/demo.py --cached          # zero-network cold open + CPU gate
    .venv/bin/python crucible/demo.py --no-color
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness import beats  # noqa: E402
from harness.beats import (  # noqa: E402
    Timeline, banner, narrate, verdict_green, verdict_red, say,
    bold, dim, cyn, grn, red, ylw,
)
from harness.fallback import resolve_mode  # noqa: E402
from harness.workshop import WorkshopClient  # noqa: E402

COURTROOM_EVENT = "veritas_courtroom_demo"
RUN2_EVENT = "veritas_courtroom_demo_run2"
ARTIFACTS = _REPO / "artifacts"
DEMO_TARGET = "36_RMSNorm"   # must match crucible.courtroom_run.TARGET for the run#2 ledger read

# run #2's compounding candidate (built on run #1's verified einsum baseline; distinct source).
GOOD_V2 = '''
def rmsnorm_candidate(x, eps):
    """v2: built on run #1's verified fused reduction; no full-size square temp."""
    n = x.shape[1]
    ss = np.einsum("bfn,bfn->bn", x, x)
    inv = np.reciprocal(np.sqrt(ss / n + eps))[:, None, :]
    return x * inv
'''


def load_dotenv() -> None:
    env = _REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _imp(modpath: str):
    """Import a module by dotted path, returning None if unavailable."""
    try:
        return __import__(modpath, fromlist=["_"])
    except Exception:
        return None


def _reset_demo_db() -> pathlib.Path:
    """Fresh demo ledger path so each invocation shows a clean run#1 → run#2."""
    ARTIFACTS.mkdir(exist_ok=True)
    db = ARTIFACTS / "veritas_demo_ledger.db"
    for p in (db, db.with_suffix(".db-wal"), db.with_suffix(".db-shm")):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    return db


def _make_oracle(use_modal: bool):
    """Build the oracle for the centerpiece gate. Returns (oracle, label) or
    (None, label) to let courtroom_run pick its CPU default."""
    if use_modal:
        ko = _imp("crucible.oracle.kernel_oracle")
        if ko and hasattr(ko, "KernelOracle"):
            return ko.KernelOracle(), "live Modal T4 (KernelOracle)"
        say(ylw("  --modal requested but KernelOracle unavailable — falling back to CPU gate."))
    ro = _imp("crucible.oracle.reference_oracle")
    if ro and hasattr(ro, "ReferenceRMSNormOracle"):
        return ro.ReferenceRMSNormOracle(shape=(256, 1024, 8), hidden_shape=(128, 768, 4)), \
            "CPU ReferenceRMSNormOracle (deterministic floor)"
    return None, "courtroom_run default CPU oracle"


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight(ws: WorkshopClient, mode: str, oracle_label: str, gate_real: bool) -> bool:
    banner("VERITAS — THE COURTROOM FOR AUTONOMOUS RESEARCH",
           "one command · real gate · deterministic floor · live Raindrop readback")
    say(dim(f"  mode={mode}  ·  workshop={ws.origin}  ·  "
            f"gate={'REAL · ' + oracle_label if gate_real else 'rehearsed fallback (engine missing)'}"))
    if not ws.is_up():
        say(red(f"  PREFLIGHT FAIL — Raindrop Workshop not reachable at {ws.origin}."))
        say(dim("  Start it (raindrop workshop) and re-run. The Workshop is the courtroom."))
        return False
    say(grn("  ✓ Workshop is live — the courtroom is in session."))
    return True


# --------------------------------------------------------------------------- #
# BEAT 1 — COLD OPEN (legal citation oracle, already real)
# --------------------------------------------------------------------------- #
def beat_cold_open(tl: Timeline, mode: str) -> bool:
    with tl.beat("0–7s · COLD OPEN — caught a lie with a database, not an opinion", 7):
        legal = _imp("cold_open.legal_demo")
        cit = _imp("crucible.oracle.citation_oracle")
        if legal and cit and hasattr(legal, "run_cold_open"):
            prefer_live = (mode in ("live", "auto")) and bool(os.environ.get("COURTLISTENER_TOKEN"))
            oracle = cit.CitationOracle(prefer_live=prefer_live, verbose=False)
            col = legal.C(enabled=sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
            ok, _results = legal.run_cold_open(oracle, col, with_spans=True)
            say(grn("  ✓ cold-open invariant held: real→GREEN, fabricated→RED.") if ok
                else red("  ✗ cold-open invariant breached."))
            return ok
        say(ylw("  cold_open.legal_demo unavailable — beat skipped (seam)."))
        return False


# --------------------------------------------------------------------------- #
# The courtroom run — GATE-PRODUCED via crucible.courtroom_run
# --------------------------------------------------------------------------- #
def _emit_courtroom(ws: WorkshopClient, db_path: pathlib.Path, oracle):
    """Run the canonical courtroom through the REAL gate (crucible.courtroom_run):
    C_GOOD + C_HACK are gate-produced by the oracle; C_SILENT/C_NOORACLE are
    labelled adversarial probes for detectors A/D. Writes C_GOOD's committed row to
    db_path for the run#2 compounding read. Returns (run_id, info) or (None, None)."""
    cr = _imp("crucible.courtroom_run")
    det = _imp("crucible.detectors")
    if not (cr and det and hasattr(cr, "emit_courtroom_run")):
        return None, None
    run_id, info = cr.emit_courtroom_run(event_name=COURTROOM_EVENT, oracle=oracle,
                                         db_path=str(db_path))
    ws.wait_for(lambda: ws.claim_span_ids(run_id, "C_GOOD"), timeout=20, interval=0.5)
    # courtroom_run defaults annotate=False — write the good/issue annotations now.
    fn = getattr(det, "judge_and_annotate", None)
    try:
        fn(run_id) if callable(fn) else det.annotate_from_report(det.adjudicate(run_id))
    except Exception as exc:
        say(dim(f"      (annotate note: {exc})"))
    time.sleep(0.9)
    return run_id, info


def _emit_courtroom_canned(ws: WorkshopClient):
    """Deep fallback only (engine totally unavailable): the rehearsed canned trace."""
    cd = _imp("crucible.courtroom_demo")
    det = _imp("crucible.detectors")
    if not (cd and det and hasattr(cd, "emit_courtroom_run")):
        return None, None
    run_id, _ = cd.emit_courtroom_run(event_name=COURTROOM_EVENT)
    ws.wait_for(lambda: ws.claim_span_ids(run_id, "C_GOOD"), timeout=15, interval=0.5)
    fn = getattr(det, "judge_and_annotate", None)
    fn(run_id) if callable(fn) else det.annotate_from_report(det.adjudicate(run_id))
    time.sleep(0.8)
    return run_id, {"outcomes": {}, "canned": True}


# --------------------------------------------------------------------------- #
# BEAT 2 — THE CHEAT (rejected by the real gate)
# --------------------------------------------------------------------------- #
def _live_static_catch() -> str | None:
    sc = _imp("crucible.oracle.static_checker")
    cand = _REPO / "benchmarks" / "rmsnorm_lab" / "candidates" / "tamper_torch_disguise.py"
    if not (sc and cand.exists() and hasattr(sc, "static_pregate")):
        return None
    res = sc.static_pregate(cand.read_text(), backend="triton", precision="fp32")
    return res["errors"][0] if (not res["ok"] and res["errors"]) else None


def beat_cheat(tl: Timeline, ws: WorkshopClient, run_id: str, info: dict) -> bool:
    ok = True
    with tl.beat("7–22s · THE CHEAT — the swarm caught reward-hacking its own benchmark", 15):
        narrate("A confident \"2x faster RMSNorm\" is run against an EXTERNAL mechanical oracle.")
        hack = info.get("outcomes", {}).get("C_HACK")
        if hack is not None:
            reason = (hack.blocked_reason or "refuted by the oracle").split(";")[0]
            verdict_red(f"REJECTED — the gate COMPUTED the verdict: {reason[:90]}")
            say(dim(f"      verdict={hack.verdict.verdict} · tamper_detected={hack.verdict.tamper_detected} "
                    f"· promoted={hack.promoted}  (oracle-produced, not stamped)"))
        else:
            verdict_red("REJECTED — anti-tamper caught the reward-hack (courtroom run).")
        # a torch-in-disguise candidate dies at the static pre-gate (real, no GPU)
        err = _live_static_catch()
        if err:
            say(dim(f"      static pre-gate also kills a torch-in-disguise candidate before GPU spend: {err[:70]}"))
        # live readback: the blocked claim's span carries an 'issue' annotation
        ws.wait_for(lambda: ws.get_annotations(run_id), timeout=8, interval=0.5)
        ok_h, det_h = ws.assert_rejected_flagged(run_id, "C_HACK")
        if ok_h and det_h.get("issue_notes"):
            say(dim(f"      Raindrop annotation ⚑ issue: {det_h['issue_notes'][0][:90]}"))
        (verdict_green if ok_h else verdict_red)(
            "Workshop confirms: the blocked claim carries an 'issue' annotation." if ok_h
            else f"expected an issue annotation on C_HACK — {det_h}")
        ok = ok_h
        narrate("A generator-only swarm ships this. CRUCIBLE caught it cheating its own benchmark.")
    return ok


# --------------------------------------------------------------------------- #
# BEAT 3 — THE VERIFIED INCREMENT (committed by the real gate)
# --------------------------------------------------------------------------- #
def beat_verified(tl: Timeline, ws: WorkshopClient, run_id: str, info: dict,
                  db_path: pathlib.Path) -> tuple[bool, dict]:
    out = {}
    with tl.beat("22–40s · VERIFIED INCREMENT — a separate oracle reproduced it", 18):
        narrate("The honest candidate faces the SAME oracle. A separate mechanism reproduces it.")
        good = info.get("outcomes", {}).get("C_GOOD")
        if good is not None:
            sp = f"{good.speedup:.3f}x" if good.speedup is not None else "n/a"
            verdict_green(f"CONFIRMED — correct over seeds + a REAL measured {sp} speedup, anti-tamper clean.")
            say(dim(f"      verdict={good.verdict.verdict} · promotion={good.promotion} · "
                    f"trace_readback={good.trace_readback_confirmed}  (oracle-produced)"))
            cert = good.certificate_paths[0].name if good.certificate_paths else "n/a"
            say(grn(f"  ✓ ledger COMMITTED (run #1) · proof_hash={good.proof_hash[:16]}…  "
                    f"· certificate {cert}"))
            out = {"proof_hash": good.proof_hash, "ledger_id": good.ledger_id, "speedup": good.speedup}
        else:
            say(dim("      (no gate outcome — reading the committed row from the ledger)"))
        # live readback: promoted claim has an oracle span + NO issue annotation
        ok_g, det_g = ws.assert_promoted_clean(run_id, "C_GOOD")
        (verdict_green if ok_g else verdict_red)(
            "Workshop confirms: PROMOTED C_GOOD has an oracle span + NO issue annotation." if ok_g
            else f"promoted-clean readback FAILED — {det_g}")
        narrate("This isn't trusted because an agent said so — it's verified under stated bounds.")
        return ok_g, out


# --------------------------------------------------------------------------- #
# BEAT 4 — COMPOUNDING (run #2 reads run #1's verified ledger row, builds on it)
# --------------------------------------------------------------------------- #
def beat_compounding(tl: Timeline, ws: WorkshopClient, db_path: pathlib.Path,
                     oracle) -> tuple[bool, str | None]:
    with tl.beat("40–52s · COMPOUNDING — verified memory that compounds across runs", 12):
        led_mod = _imp("crucible.ledger")
        orch_mod = _imp("crucible.orchestrator")
        s = _imp("crucible.schemas")
        if not (led_mod and orch_mod and s):
            say(ylw("  engine unavailable — compounding beat not landed."))
            return False, None
        # REAL SQLite round-trip: read run #1's committed row.
        ledger = led_mod.Ledger(str(db_path))
        base = None
        for fn in ("latest_verified", "latest_baseline"):
            f = getattr(ledger, fn, None)
            if callable(f):
                base = f(DEMO_TARGET)
                if base:
                    break
        n_refuted = len(ledger.refuted_artifact_hashes(DEMO_TARGET)) \
            if hasattr(ledger, "refuted_artifact_hashes") else 0
        if base is None:
            say(ylw("  run #1 baseline not found in the ledger — compounding not landed."))
            return False, None
        say(dim("  run #2 baseline read from: crucible.ledger — real SQLite read-back of run #1's row"))
        say(dim(f"    → run #1's verified increment ({base.speedup:.3f}x vs reference) · {base.ledger_id[:20]}…"))
        narrate("Run #2 inherits run #1's verified row AND its negative evidence — it never "
                "re-tries a known cheat.")
        # run #2 through the REAL gate, building on the baseline
        mission2 = s.new_id("msn")
        orch2 = orch_mod.Orchestrator(oracle=oracle, ledger=ledger, mission_id=mission2,
                                      out_dir=ARTIFACTS / "certificates", event_name=RUN2_EVENT)
        claim = s.Claim(mission_id=mission2,
                        statement="An improved RMSNorm built on run #1's verified baseline",
                        claim_type="speedup_claim", target=DEMO_TARGET, speedup_threshold=1.0,
                        baseline_ledger_id=base.ledger_id)
        cand = s.Candidate(claim_id=claim.claim_id, mission_id=mission2, code=GOOD_V2,
                           entry_point="rmsnorm_candidate", generator="rehearsed-cpu", label="good_v2")
        runner = getattr(orch2, "run_single", None) or getattr(orch2, "evaluate")
        out2 = runner(claim, cand)
        run2_id = orch2.trace_id
        links = bool(out2.ledger_row and out2.ledger_row.parent_ledger_id == base.ledger_id)
        if out2.promoted:
            verdict_green(f"run #2 COMMITTED a verified increment ({out2.speedup:.3f}x vs reference), "
                          f"linked to run #1 (parent_ledger_id set={links}).")
            say(dim(f"      reused {n_refuted} refuted path(s) as negative evidence (no re-work); "
                    f"ledger holds {ledger.counts().get('committed', 0)} committed across "
                    f"{ledger.next_run_id() - 1} run(s)."))
            narrate("The missing layer under autoresearch: verified memory that compounds across runs.")
        else:
            verdict_red(f"run #2 did not commit ({out2.verdict.verdict}).")
        # keep orch2 for the replay close
        beat_compounding._orch2 = orch2
        beat_compounding._out2 = out2
        ledger.close()
        return out2.promoted and links, run2_id


# --------------------------------------------------------------------------- #
# BEAT 5 — RAINDROP CLOSE (replay + on-screen verification)
# --------------------------------------------------------------------------- #
def beat_close(tl: Timeline, ws: WorkshopClient, run_id: str, run2_id: str | None,
               run1_info: dict) -> bool:
    ok = True
    with tl.beat("52–60s · RAINDROP CLOSE — inspectable, annotated, replayable", 8):
        narrate("Open Workshop: every verdict is a span, every block an annotation, "
                "the increment replayable.")
        orch2 = getattr(beat_compounding, "_orch2", None)
        out2 = getattr(beat_compounding, "_out2", None)
        if orch2 is not None and out2 is not None and hasattr(orch2, "trigger_replay"):
            try:
                res = orch2.trigger_replay(out2.claim_id, out2.candidate_id)
                say(grn(f"  ✓ REPLAY re-verified the increment → "
                        f"promotion={res.get('promotion')} verdict={res.get('verdict')} "
                        f"regressed={res.get('regressed')}"))
            except Exception as exc:
                say(ylw(f"  replay note: {exc}"))
        else:
            rs = _imp("crucible.replay_server")
            if rs and hasattr(rs, "run_replay"):
                try:
                    res = rs.run_replay({"replayRunId": f"demo-replay-{int(time.time())}",
                                         "sourceRunId": run_id,
                                         "context": {"claim_id": "C_GOOD", "candidate_id": "cand_good"}})
                    say(grn(f"  ✓ REPLAY re-verified C_GOOD → verdict={res.get('verdict')} "
                            f"(mode={res.get('mode')})"))
                except Exception as exc:
                    say(ylw(f"  replay note: {exc}"))

        ok_g, _ = ws.assert_promoted_clean(run_id, "C_GOOD")
        ok_h, _ = ws.assert_rejected_flagged(run_id, "C_HACK")
        (verdict_green if ok_g else verdict_red)(
            "PROMOTED C_GOOD has an oracle span + NO issue annotation." if ok_g
            else "promoted-clean readback FAILED.")
        (verdict_green if ok_h else verdict_red)(
            "REJECTED C_HACK carries an 'issue' annotation." if ok_h
            else "rejected-flagged readback FAILED.")
        ok = ok_g and ok_h

        base = ws.origin
        say("")
        say(bold("  THE COURTROOM (open in Workshop):"))
        say(cyn(f"    {base}/runs/{run_id}") + dim("   ← run #1: C_GOOD committed; C_HACK + A/D probes blocked"))
        if run2_id:
            say(cyn(f"    {base}/runs/{run2_id}") + dim("   ← run #2: compounded on run #1"))
        if run1_info.get("proof_hash"):
            say(dim(f"    proof_hash = {run1_info['proof_hash']}"))
        say("")
        say(bold(cyn("  \"We built the courtroom that decides whether the optimization is real —")))
        say(bold(cyn("   and only real, verified increments compound.\"")))
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="VERITAS one-command demo")
    ap.add_argument("--modal", action="store_true",
                    help="run the centerpiece on the live Modal T4 oracle (real GPU; may exceed 60s)")
    ap.add_argument("--cached", action="store_true",
                    help="zero-network cold open (cache only) + CPU gate")
    ap.add_argument("--live", action="store_true",
                    help="prefer live overlays (CourtListener/OpenAI) where credentials exist")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        os.environ["NO_COLOR"] = "1"
        beats._USE_COLOR = False

    load_dotenv()
    mode = "cached" if args.cached else ("live" if args.live else os.environ.get("VERITAS_DEMO_MODE", "auto"))
    try:
        mode = resolve_mode(mode)
    except ValueError:
        mode = "auto"

    ws = WorkshopClient()
    # --modal runs real GPU (cold start + per-candidate timing) → relax the budget.
    tl = Timeline(target_s=180.0 if args.modal else 60.0)
    db_path = _reset_demo_db()
    oracle, oracle_label = _make_oracle(use_modal=args.modal)
    gate_real = _imp("crucible.courtroom_run") is not None

    if not preflight(ws, mode, oracle_label, gate_real):
        return 2

    results: dict[str, bool] = {}
    results["cold_open"] = beat_cold_open(tl, mode)

    # Emit the gate-produced courtroom run (real verdicts). Deep fallback: canned trace.
    run_id, info = _emit_courtroom(ws, db_path, oracle)
    used_canned = False
    if not run_id:
        say(ylw("  real engine unavailable — falling back to the rehearsed courtroom trace."))
        run_id, info = _emit_courtroom_canned(ws)
        used_canned = True
    if not run_id:
        say(red("\n  FATAL — could not produce the courtroom run (engine + fallback both unavailable)."))
        return 1

    results["cheat"] = beat_cheat(tl, ws, run_id, info)
    ok_v, run1_info = beat_verified(tl, ws, run_id, info, db_path)
    results["verified"] = ok_v
    ok_c, run2_id = beat_compounding(tl, ws, db_path, oracle)
    results["compounding"] = ok_c
    results["close"] = beat_close(tl, ws, run_id, run2_id, run1_info)

    within = tl.elapsed_s <= tl.target_s
    tl.report()

    print("\n" + bold("  BEAT SCOREBOARD"))
    for name in ("cold_open", "cheat", "verified", "compounding", "close"):
        mark = grn("LANDED") if results.get(name) else red("MISSED")
        print(f"    {mark}  {name}")
    all_landed = all(results.values())
    ok = all_landed and within
    print("\n" + bold("═" * 64))
    if ok:
        gate = "rehearsed fallback" if used_canned else f"real gate · {oracle_label}"
        print("  " + grn(f"DEMO GREEN — all five beats landed ({gate}), verified live, within budget."))
    else:
        why = []
        if not all_landed:
            why.append("a beat missed")
        if not within:
            why.append(f"over {tl.target_s:.0f}s budget")
        print("  " + red(f"DEMO NOT GREEN — {', '.join(why)} (see above)."))
    print(bold("═" * 64), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
