#!/usr/bin/env python
"""Freeze the live oracle verdicts into per-candidate JSON for the no-Modal demo fallback
(FLOOR §1: cached artifacts so a Modal hiccup can't kill the run).

Parses the captured run in modal/logs/oracle-proof.log and writes
benchmarks/rmsnorm_lab/cached_verdicts/<candidate_id>.json — the exact JSON the deployed
verifier returned. demo-verifier loads these directly when running without Modal.

Usage:  .venv/bin/python modal/freeze_cached_verdicts.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "modal" / "logs" / "oracle-proof.log"
OUT = ROOT / "benchmarks" / "rmsnorm_lab" / "cached_verdicts"


def extract_verdicts(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            for j in range(i, n):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[i:j + 1])
                            if isinstance(obj, dict) and "oracle_type" in obj and "candidate_id" in obj:
                                out[obj["candidate_id"]] = obj
                        except json.JSONDecodeError:
                            pass
                        i = j
                        break
        i += 1
    return out


def main() -> int:
    if not LOG.exists():
        print(f"no log at {LOG} — run modal/run_oracle_proof.py --selftest first")
        return 2
    verdicts = extract_verdicts(LOG.read_text())
    if not verdicts:
        print("no verdicts found in log")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    for cid, v in sorted(verdicts.items()):
        path = OUT / f"{cid}.json"
        path.write_text(json.dumps(v, indent=2))
        index[cid] = {
            "verdict": v.get("verdict"), "correctness_passed": v.get("correctness_passed"),
            "tamper_detected": v.get("tamper_detected"), "speedup": v.get("speedup"),
            "blocked_reason": v.get("blocked_reason"), "file": path.name,
        }
        print(f"froze {cid:<24} -> {path.relative_to(ROOT)}  ({v.get('verdict')})")
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(verdicts)} cached verdicts + index.json to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
