"""crucible/oracle/reference_oracle.py — a REAL CPU execution oracle (numpy).

This is NOT a stub and NOT modal-oracle's GPU ``kernel_oracle`` — it is an
independent, fully-functional execution oracle used so the CRUCIBLE SPINE has its
own end-to-end acceptance (FLOOR §3.A) on any machine, with no GPU / Modal / API
key.  It genuinely:

  * runs a static pre-gate (regex) that blocks torch-in-disguise / try-except / bare-pass
  * exec's the candidate source in a restricted namespace and calls its entry point
  * checks correctness vs a reference RMSNorm over 5 seeds + a HIDDEN extra shape/seed
    (torch.allclose-equivalent np.allclose at atol=rtol=1e-2), recomputing the reference
    from a PRISTINE copy of the inputs
  * runs the named FLOOR §2.2 anti-tamper defenses, adapted to CPU:
      - input-mutation detection (zero-inputs cheat)
      - input-sensitivity / result-reuse detection (stale-buffer cheat)
      - output materialization + isnan/isinf (uninitialized-buffer cheat)
      - >10x excessive-speedup rejection (timing fraud)
      - dual-timer disagreement (>1.5x between perf_counter and process_time, above a
        noise floor so honest sub-millisecond kernels are never falsely flagged)

The GPU production path (modal-oracle) measures the same things with cuda_event /
do_bench / Triton on a real Modal sandbox; both return the same
:class:`crucible.schemas.Verdict` and pass through the SAME truth floor.

Candidate contract: ``candidate.code`` defines a callable named
``candidate.entry_point`` (default ``"rmsnorm_candidate"``) with signature
``(x: np.ndarray, eps: float) -> np.ndarray`` computing RMSNorm reduced over
``reduce_axis`` (default 1), eps inside the sqrt, no weight.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

import numpy as np

from crucible.schemas import (
    AntiTamperResult,
    Assumptions,
    Candidate,
    Claim,
    CorrectnessResult,
    OracleProtocol,
    SpeedResult,
    Verdict,
)

# --- static pre-gate patterns (CPU analog of static_checker.py) ------------- #
# These die BEFORE execution — a disguise/fallback cheat never gets to run.
_FORBIDDEN_PATTERNS = [
    (r"\btry\b", "try/except fallback (code_bypass) is forbidden"),
    (r"\bexcept\b", "try/except fallback (code_bypass) is forbidden"),
    (r"^\s*pass\s*$", "bare `pass` body (inheritance bypass) is forbidden"),
    (r"\brms_norm\b", "torch-in-disguise: rms_norm reference call is forbidden"),
    (r"_builtin_rms", "torch-in-disguise: builtin reference call is forbidden"),
    (r"\bimport\s+os\b", "os import is forbidden in a candidate kernel"),
    (r"\bimport\s+sys\b", "sys import is forbidden in a candidate kernel"),
    (r"\bsubprocess\b", "subprocess is forbidden in a candidate kernel"),
    (r"__import__", "dynamic __import__ is forbidden in a candidate kernel"),
]

_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "range", "len", "float", "int", "abs", "min", "max", "sum", "pow",
        "enumerate", "zip", "map", "list", "tuple", "dict", "bool", "round",
        "True", "False", "None", "print",
    )
}

DUAL_TIMER_THRESHOLD = 1.5
DUAL_TIMER_NOISE_FLOOR_S = 5e-4   # below this, timing is noise — do not flag
EXCESSIVE_SPEEDUP_THRESHOLD = 10.0


def static_pregate(code: str) -> tuple[bool, list[str]]:
    """Return (ok, reasons).  ok=False means block before execution."""
    reasons = []
    for pattern, why in _FORBIDDEN_PATTERNS:
        if re.search(pattern, code, flags=re.MULTILINE):
            reasons.append(why)
    return (not reasons), reasons


def _compile_candidate(code: str, entry_point: str) -> Callable:
    """Exec the candidate source in a restricted namespace and return its entry
    point.  Raises ValueError if the candidate is unusable (a *candidate* fault,
    surfaced by the oracle as a refuted verdict — not a verifier error)."""
    ns: dict = {"__builtins__": _SAFE_BUILTINS, "np": np, "numpy": np}
    try:
        exec(compile(code, "<candidate>", "exec"), ns)  # noqa: S102 — sandboxed self-test fixtures
    except Exception as exc:  # candidate didn't compile/run top-level
        raise ValueError(f"candidate source failed to load: {type(exc).__name__}: {exc}") from exc
    fn = ns.get(entry_point)
    if not callable(fn):
        raise ValueError(f"candidate does not define a callable {entry_point!r}")
    return fn


class ReferenceRMSNormOracle:
    """A real CPU RMSNorm execution oracle (the spine's self-test verifier)."""

    name = "reference_rmsnorm_cpu"

    def __init__(
        self,
        *,
        reduce_axis: int = 1,
        eps: float = 1e-5,
        shape: tuple[int, ...] = (64, 512, 8),
        hidden_shape: tuple[int, ...] = (32, 768, 4),
        atol: float = 1e-2,
        rtol: float = 1e-2,
        trials: int = 5,
        warmup: int = 3,
        timing_trials: int = 30,
        base_seed: int = 42,
        hidden_seed: int = 1337,
        dual_timer: bool = False,
    ):
        self.reduce_axis = reduce_axis
        self.eps = eps
        self.shape = shape
        self.hidden_shape = hidden_shape
        self.atol = atol
        self.rtol = rtol
        self.trials = trials
        self.warmup = warmup
        self.timing_trials = timing_trials
        self.base_seed = base_seed
        self.hidden_seed = hidden_seed
        # The wall-vs-cpu (perf_counter vs process_time) dual timer is a CPU proxy
        # for the GPU cuda_event-vs-do_bench stream-bypass check. On CPU, wall >> cpu
        # is ordinary OS scheduling — NOT a timing bypass — so it false-positives on
        # honest candidates under contention. OFF by default here; the GPU KernelOracle
        # keeps the real dual timer where it's meaningful.
        self.dual_timer = dual_timer

    # reference forward (the mechanical truth): RMSNorm over reduce_axis, eps in sqrt, no weight.
    # Accepts an optional eps so it is call-compatible with candidates: fn(x, eps).
    def reference(self, x: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
        e = self.eps if eps is None else eps
        ms = np.mean(np.square(x), axis=self.reduce_axis, keepdims=True)
        return x / np.sqrt(ms + e)

    def _input(self, seed: int, shape: tuple[int, ...]) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape, dtype=np.float64).astype(np.float64)

    def _time(self, fn: Callable, x: np.ndarray) -> tuple[float, float]:
        """Return (perf_counter_seconds, process_time_seconds) as the MIN over trials.

        Min (best) time is the standard micro-benchmark statistic: it reflects the
        uncontended compute cost, so the measured speedup is stable run-to-run.
        Median is contaminated by OS scheduling contention, which on CPU can make a
        genuinely-faster candidate measure marginally slower (a flaky sub-1.0x dip)."""
        for _ in range(self.warmup):
            fn(x.copy(), self.eps)
        wall, cpu = [], []
        for _ in range(self.timing_trials):
            w0, c0 = time.perf_counter(), time.process_time()
            fn(x.copy(), self.eps)
            wall.append(time.perf_counter() - w0)
            cpu.append(time.process_time() - c0)
        return float(min(wall)), float(min(cpu))

    def verify(self, claim: Claim, candidate: Candidate) -> Verdict:
        seeds = [self.base_seed + i for i in range(self.trials)]
        assumptions = Assumptions(
            shape=f"{self.shape} + hidden {self.hidden_shape}",
            dtype="float64",
            hardware="CPU (numpy reference oracle)",
            tolerance=f"np.allclose atol={self.atol} rtol={self.rtol}",
            seeds=seeds + [self.hidden_seed],
        )
        protocol = OracleProtocol(
            correctness=f"{self.trials} trials (seeds from {self.base_seed}) + hidden shape/seed; "
                        f"np.allclose atol={self.atol} rtol={self.rtol}; candidate on cloned inputs, "
                        f"reference from pristine copy; shape/isnan/isinf asserted.",
            speed=f"{self.warmup} warmup + {self.timing_trials} timed trials (min, uncontended); "
                  f"speedup = reference_time / candidate_time.",
            anti_tamper=f"input-mutation + input-sensitivity (result-reuse) + materialization "
                        f"(isnan/isinf) + >{EXCESSIVE_SPEEDUP_THRESHOLD}x excessive-speedup"
                        + (f" + dual-timer >{DUAL_TIMER_THRESHOLD}x disagreement" if self.dual_timer
                           else " (wall-vs-cpu dual-timer is GPU-path; off on CPU)") + ".",
        )

        def _base_verdict(**kw) -> Verdict:
            return Verdict(
                claim_id=claim.claim_id, candidate_id=candidate.candidate_id,
                mission_id=claim.mission_id, oracle_type="kernel",
                hardware="CPU (numpy reference oracle)", assumptions=assumptions,
                oracle_protocol=protocol, **kw,
            )

        code = candidate.code or ""
        if not code:
            return _base_verdict(
                verdict="refuted", correctness_passed=False, tamper_detected=False,
                blocked_reason="candidate has no code to execute",
            )

        # --- 1. STATIC PRE-GATE (before any execution) --------------------- #
        ok, static_reasons = static_pregate(code)
        if not ok:
            reason = "; ".join(static_reasons)
            return _base_verdict(
                verdict="blocked", correctness_passed=False, tamper_detected=True,
                static_check_passed=False, blocked_reason=f"static pre-gate: {reason}",
                anti_tamper=AntiTamperResult(
                    tamper_detected=True, static_check_passed=False, reasons=static_reasons,
                ),
            )

        # --- 2. COMPILE + LOAD the candidate ------------------------------- #
        entry = candidate.entry_point or "rmsnorm_candidate"
        try:
            fn = _compile_candidate(code, entry)
        except ValueError as exc:  # candidate fault -> refuted (verifier is fine)
            return _base_verdict(
                verdict="refuted", correctness_passed=False, tamper_detected=False,
                static_check_passed=True, blocked_reason=str(exc),
            )

        # --- everything below can surface a VERIFIER error (loud, never hidden) --
        try:
            tamper_reasons: list[str] = []
            mutated = False
            insensitive = False
            non_materialized = False
            mismatches: list[str] = []
            max_abs_err = 0.0
            outputs: list[np.ndarray] = []

            all_shapes = [(s, self.shape) for s in seeds]
            all_shapes.append((self.hidden_seed, self.hidden_shape))  # HIDDEN extra seed+shape

            for seed, shape in all_shapes:
                x = self._input(seed, shape)
                pristine = x.copy()
                work = x.copy()
                out = np.asarray(fn(work, self.eps))

                # anti-tamper: did the candidate mutate the input it was handed?
                if work.shape == pristine.shape and not np.array_equal(work, pristine):
                    mutated = True
                    tamper_reasons.append(f"seed {seed}: candidate mutated its input in place")

                ref = self.reference(pristine)
                # materialization: real, finite, correctly-shaped output
                if out.shape != ref.shape:
                    non_materialized = True
                    mismatches.append(f"seed {seed}: output shape {out.shape} != reference {ref.shape}")
                    outputs.append(out)
                    continue
                if not np.all(np.isfinite(out)):
                    non_materialized = True
                    tamper_reasons.append(f"seed {seed}: output has NaN/Inf (non-materialized / reused buffer)")
                    outputs.append(out)
                    continue

                err = float(np.max(np.abs(out - ref)))
                max_abs_err = max(max_abs_err, err)
                if not np.allclose(out, ref, atol=self.atol, rtol=self.rtol):
                    mismatches.append(f"seed {seed}: max_abs_err={err:.3e} exceeds tolerance")
                outputs.append(out)

            # anti-tamper: input-sensitivity / result-reuse — different inputs that
            # produce identical outputs while the reference differs = stale buffer.
            finite_pairs = [
                (i, j) for i in range(len(outputs)) for j in range(i + 1, len(outputs))
                if outputs[i].shape == outputs[j].shape
            ]
            for i, j in finite_pairs:
                if np.array_equal(outputs[i], outputs[j]):
                    insensitive = True
                    tamper_reasons.append(
                        "candidate produced identical output for distinct inputs (result-reuse / input-insensitive)"
                    )
                    break

            correctness_passed = (not mismatches) and (not non_materialized)

            # --- 3. SPEED + dual-timer + excessive-speedup ----------------- #
            x_speed = self._input(self.base_seed, self.shape)
            ref_wall, ref_cpu = self._time(self.reference, x_speed)
            cand_wall, cand_cpu = self._time(fn, x_speed)
            speedup = (ref_wall / cand_wall) if cand_wall > 0 else float("inf")

            # dual-timer disagreement — GPU-only (off by default on CPU; see __init__).
            # On CPU, wall>>cpu is OS scheduling, not a timing bypass, so enabling it
            # here false-positives on honest candidates. Computed for evidence only
            # unless self.dual_timer is explicitly set.
            dual_ratio = None
            dual_disagree = False
            if self.dual_timer and cand_wall >= DUAL_TIMER_NOISE_FLOOR_S and cand_cpu >= DUAL_TIMER_NOISE_FLOOR_S:
                lo, hi = sorted((cand_wall, cand_cpu))
                dual_ratio = (hi / lo) if lo > 0 else float("inf")
                dual_disagree = dual_ratio > DUAL_TIMER_THRESHOLD
                if dual_disagree:
                    tamper_reasons.append(
                        f"dual-timer disagreement {dual_ratio:.2f}x (wall {cand_wall*1e3:.3f}ms vs "
                        f"cpu {cand_cpu*1e3:.3f}ms) — timing bypass"
                    )

            excessive = bool(speedup is not None and speedup > EXCESSIVE_SPEEDUP_THRESHOLD)
            if excessive:
                tamper_reasons.append(f"excessive speedup {speedup:.1f}x > {EXCESSIVE_SPEEDUP_THRESHOLD}x — timing fraud")

            tamper_detected = bool(mutated or insensitive or non_materialized or dual_disagree or excessive)

            anti = AntiTamperResult(
                tamper_detected=tamper_detected,
                static_check_passed=True,
                dual_timer_ratio=dual_ratio,
                excessive_speedup=excessive,
                reasons=tamper_reasons,
            )
            corr = CorrectnessResult(
                passed=correctness_passed, trials=len(all_shapes), seeds=seeds + [self.hidden_seed],
                atol=self.atol, rtol=self.rtol, max_abs_err=max_abs_err, mismatches=mismatches,
            )
            spd = SpeedResult(
                speedup=speedup, ref_time_ms=ref_wall * 1e3, candidate_time_ms=cand_wall * 1e3,
                warmup=self.warmup, trials=self.timing_trials,
                threshold=(claim.effective_threshold if claim.is_speedup_claim else None),
            )

            # --- 4. ASSEMBLE the verdict ----------------------------------- #
            if tamper_detected:
                verdict, reason = "refuted", "; ".join(tamper_reasons)
            elif not correctness_passed:
                verdict, reason = "refuted", "; ".join(mismatches) or "correctness failed"
            else:
                verdict, reason = "confirmed", None

            return _base_verdict(
                verdict=verdict,
                correctness_passed=correctness_passed,
                tamper_detected=tamper_detected,
                static_check_passed=True,
                speedup=speedup,
                speedup_threshold=(claim.effective_threshold if claim.is_speedup_claim else None),
                blocked_reason=reason,
                correctness=corr,
                speed=spd,
                anti_tamper=anti,
                evidence={
                    "ref_time_ms": ref_wall * 1e3, "candidate_time_ms": cand_wall * 1e3,
                    "ref_cpu_ms": ref_cpu * 1e3, "candidate_cpu_ms": cand_cpu * 1e3,
                    "max_abs_err": max_abs_err,
                },
            )
        except Exception as exc:  # the VERIFIER itself broke — surface as ERROR, never hide
            from crucible.oracle.base import make_error_verdict
            return make_error_verdict(claim, candidate, exc, oracle_type="kernel")


__all__ = ["ReferenceRMSNormOracle", "static_pregate"]
