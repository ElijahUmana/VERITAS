#!/usr/bin/env python
"""VERITAS oracle PROOF — deploy-once, call-many; verify the honest candidate and catch every cheat.

Usage (after `modal deploy modal/verifier_app.py`):
    .venv/bin/python modal/run_oracle_proof.py            # all candidates
    .venv/bin/python modal/run_oracle_proof.py --selftest # + image selftest first
    .venv/bin/python modal/run_oracle_proof.py good_rehearsed tamper_stream   # subset

This is the "actually verify it" gate: it asserts each candidate gets the EXACT verdict its
named defense should produce. Exit 0 only if every expectation holds. A candidate that errors is
surfaced loudly (REFUTED), never a silent pass.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from crucible.oracle import kernel_oracle  # noqa: E402

# Candidate -> the verdict its NAMED defense must produce.
EXPECT = {
    "good_rehearsed":        {"verdict": "confirmed", "correctness_passed": True,  "tamper_detected": False},
    "wrong_lastdim":         {"verdict": "refuted",   "correctness_passed": False, "tamper_detected": False},
    "tamper_result_reuse":   {"verdict": "refuted",   "tamper_detected": True},
    "tamper_zero_inputs":    {"verdict": "refuted",   "tamper_detected": True},
    "tamper_stream":         {"verdict": "refuted",   "tamper_detected": True},
    "tamper_torch_disguise": {"verdict": "blocked",   "tamper_detected": True},
}
DEFENSE = {
    "good_rehearsed": "honest -> CONFIRMED (correct on 5 seeds + hidden, real BW-bound speedup, timers agree)",
    "wrong_lastdim": "correctness oracle (allclose) -> REFUTED (honest mistake, no tamper)",
    "tamper_result_reuse": "output materialization (poison + isnan) -> REFUTED + tamper",
    "tamper_zero_inputs": "pristine-copy recompute (clone inputs) -> REFUTED + tamper",
    "tamper_stream": "DUAL timer disagreement (cuda_event vs do_bench) -> REFUTED + tamper",
    "tamper_torch_disguise": "static pre-gate (no GPU spend) -> BLOCKED + tamper",
}


def _check(name: str, v: dict) -> tuple[bool, list[str]]:
    fails = []
    for k, want in EXPECT[name].items():
        got = v.get(k)
        if got != want:
            fails.append(f"{k}: expected {want!r}, got {got!r}")
    # An honest confirmed claim must show a real (>1x) measured speedup.
    if name == "good_rehearsed":
        sp = v.get("speedup")
        if sp is None or sp <= 1.0:
            fails.append(f"speedup: expected a real >1.0x win, got {sp!r}")
    # The disguise cheat must never reach the GPU.
    if name == "tamper_torch_disguise" and v.get("hardware") is not None:
        fails.append("disguise reached the GPU (hardware set) — should be blocked client-side")
    return (len(fails) == 0), fails


# Reusable-core probe: a candidate that monkey-patches the JUDGE (torch.allclose) so every
# comparison "passes" while returning garbage. The static checker MISSES this (it's a runtime
# patch); the harness-integrity rail must catch it. Not a kernel candidate — an anti-tamper test.
INTEGRITY_PROBE = '''
import torch
import triton
import triton.language as tl

# CHEAT: patch the judge at import so the oracle's torch.allclose always returns True.
torch.allclose = lambda *a, **k: True

@triton.jit
def _k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m), mask=m)

class ModelNew(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__(); self.eps = eps
    def forward(self, x):
        return torch.empty_like(x)   # garbage; only "passes" if the judge is compromised
'''


def _run_integrity_probe() -> bool:
    print("\n" + "#" * 78)
    print("# REUSABLE-CORE PROBE: harness integrity (candidate patches torch.allclose)")
    print("# Named defense: snapshot/restore harness callables -> REFUTED + tamper")
    print("#" * 78)
    v = kernel_oracle.verify(
        {"claim_id": "proof", "claim_type": "speedup_claim"},
        {"candidate_id": "tamper_patch_allclose", "source": INTEGRITY_PROBE, "backend": "triton"},
    )
    print(json.dumps(v, indent=2))
    ok = (v.get("verdict") == "refuted" and v.get("tamper_detected") is True
          and "integrity" in (v.get("blocked_reason") or "").lower())
    print(("PASS ✅" if ok else "FAIL ❌"), "harness-integrity rail:", v.get("blocked_reason"))
    return ok


def main(argv: list[str]) -> int:
    do_selftest = "--selftest" in argv
    do_integrity = "--integrity" in argv or "--selftest" in argv
    names = [a for a in argv if not a.startswith("--")] or list(EXPECT.keys())

    if do_selftest:
        import modal
        st = modal.Function.from_name(kernel_oracle.APP_NAME, "selftest").remote()
        print("=" * 78)
        print("IMAGE SELFTEST:", json.dumps(st))
        if not st.get("cuda_available") or not st.get("triton_kernel_ok"):
            print("FAIL: image selftest did not confirm CUDA + a working Triton kernel.")
            return 2
        print("=" * 78)

    results: dict[str, dict] = {}
    table: list[str] = []
    all_ok = True
    for name in names:
        print("\n" + "#" * 78)
        print(f"# CANDIDATE: {name}")
        print(f"# Named defense: {DEFENSE.get(name, '?')}")
        print("#" * 78)
        verdict = kernel_oracle.verify(
            {"claim_id": "proof", "claim_type": "speedup_claim"},
            {"candidate_id": name, "name": name},
        )
        results[name] = verdict
        print(json.dumps(verdict, indent=2))

        ok, fails = _check(name, verdict)
        all_ok = all_ok and ok
        speed = verdict.get("speedup")
        speed_s = f"{speed:.2f}x" if isinstance(speed, (int, float)) else "-"
        status = "PASS ✅" if ok else "FAIL ❌"
        table.append(
            f"  {status}  {name:<22} verdict={verdict.get('verdict'):<10} "
            f"correct={str(verdict.get('correctness_passed')):<5} tamper={str(verdict.get('tamper_detected')):<5} "
            f"speedup={speed_s:<7} :: {verdict.get('blocked_reason') or 'ok'}"
        )
        if not ok:
            for f in fails:
                table.append(f"        - mismatch: {f}")

    if do_integrity:
        integ_ok = _run_integrity_probe()
        all_ok = all_ok and integ_ok
        table.append(
            f"  {'PASS ✅' if integ_ok else 'FAIL ❌'}  {'tamper_patch_allclose':<22} verdict=refuted    "
            f"correct=False tamper=True  speedup=-       :: harness integrity (allclose monkeypatch)"
        )

    print("\n" + "=" * 78)
    print("VERITAS ORACLE PROOF LEDGER")
    print("=" * 78)
    for row in table:
        print(row)
    print("=" * 78)
    if all_ok:
        print("RESULT: ✅ ALL EXPECTATIONS MET — honest candidate verified; all cheats caught live.")
        return 0
    print("RESULT: ❌ MISMATCH — see rows above. (This is a real finding, surfaced loudly.)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
