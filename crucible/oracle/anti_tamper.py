"""VERITAS anti-tamper primitives (worker-side, GPU).

VERITAS-original. Imports `torch` only — runs inside the Modal verifier and is importable
by crucible-core for assembling the Verdict. These are the explicit, deterministic defenses
named in FLOOR.md §2.2:

  - clone/zero pristine inputs            -> `clone_tensors`, `detect_input_mutation`
  - poison the allocator before candidate -> `poison_free_pool`        (kills result-reuse)
  - output materialization + isnan/isinf  -> `assert_materialized`     (kills result-reuse)
  - DUAL timer >1.5x disagreement         -> `dual_timer_disagreement` (kills stream bypass)
  - >10x excessive speedup                -> `excessive_speedup`       (kills timing fraud)

Nothing here swallows errors; failures are returned as explicit (ok, reason) tuples so the
harness can surface a loud REFUTED verdict.
"""
from __future__ import annotations

from typing import Any

import torch

# Defaults (FLOOR.md §2.2)
DUAL_TIMER_THRESHOLD = 1.5      # reject if the two timers disagree by more than this ratio
EXCESSIVE_SPEEDUP_THRESHOLD = 10.0  # reject any speedup above this (physically implausible)


def clone_tensors(tensors: list[Any]) -> list[Any]:
    """Deep-clone every tensor (non-tensors passed through). The candidate only ever
    touches clones; the reference is computed from an independent pristine copy."""
    out = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            out.append(t.clone())
        else:
            out.append(t)
    return out


def poison_free_pool(shape, dtype: torch.dtype, device, rounds: int = 3) -> None:
    """Fill the CUDA caching-allocator free pool with NaN blocks of the output size.

    A 'result-reuse' cheat returns torch.empty()/empty_like() hoping the recycled block still
    holds a previous correct result. By allocating NaN buffers of exactly the output shape and
    freeing them (back into PyTorch's caching pool, NOT released to the driver), the next
    same-size allocation the candidate makes is overwhelmingly likely to be a NaN block ->
    `assert_materialized` then trips on isnan. We deliberately do NOT call empty_cache()
    afterwards: we want those poisoned blocks to remain available for reuse.
    """
    torch.cuda.synchronize(device)
    junk = []
    for _ in range(max(1, rounds)):
        b = torch.empty(shape, dtype=dtype, device=device)
        b.fill_(float("nan"))
        junk.append(b)
    # free them back into the caching pool (reverse order), leaving NaN content behind
    del junk
    torch.cuda.synchronize(device)


def assert_materialized(out: Any, ref_shape, ref_dtype) -> tuple[bool, str]:
    """The output must be a real, fully-computed tensor of the right shape/dtype with no
    NaN/Inf. Catches result-reuse (uninitialized/poisoned memory) and degenerate outputs."""
    if not isinstance(out, torch.Tensor):
        return False, f"output is not a tensor (got {type(out).__name__})"
    if tuple(out.shape) != tuple(ref_shape):
        return False, f"output shape {tuple(out.shape)} != reference {tuple(ref_shape)}"
    if out.dtype != ref_dtype:
        return False, f"output dtype {out.dtype} != reference {ref_dtype}"
    if out.numel() == 0:
        return False, "output is empty"
    # full-tensor finiteness check — uninitialized/poisoned buffers trip here
    if torch.isnan(out).any().item():
        return False, "output contains NaN (non-materialized / reused uninitialized buffer)"
    if torch.isinf(out).any().item():
        return False, "output contains Inf (non-materialized / reused uninitialized buffer)"
    return True, ""


def detect_input_mutation(candidate_inputs: list[Any], pristine: list[Any]) -> tuple[bool, str]:
    """Detect whether the candidate mutated the inputs it was given (e.g. the zero-inputs cheat).
    The candidate received clones, so mutation never corrupts the reference — but a mutation is
    itself a tamper signal worth surfacing."""
    for i, (ci, pi) in enumerate(zip(candidate_inputs, pristine)):
        if isinstance(ci, torch.Tensor) and isinstance(pi, torch.Tensor):
            if ci.shape != pi.shape:
                return True, f"input[{i}] shape changed {tuple(pi.shape)} -> {tuple(ci.shape)}"
            if not torch.equal(ci, pi):
                return True, f"input[{i}] was mutated in place by the candidate"
    return False, ""


def dual_timer_disagreement(
    t_a_ms: float, t_b_ms: float, threshold: float = DUAL_TIMER_THRESHOLD
) -> tuple[bool, float]:
    """Compare two independent timers (cuda_event vs triton do_bench) of the SAME kernel.
    Honest kernels agree (ratio ~1). A stream-bypass shows ~0 on the default-stream cuda_event
    but the real time under do_bench's full sync -> large ratio. Returns (disagree, ratio)."""
    lo = min(abs(t_a_ms), abs(t_b_ms))
    hi = max(abs(t_a_ms), abs(t_b_ms))
    if lo <= 0.0:
        # one timer measured ~0 -> maximal disagreement (classic stream/elide hack)
        return True, float("inf")
    ratio = hi / lo
    return (ratio > threshold), float(ratio)


def excessive_speedup(speedup: float, threshold: float = EXCESSIVE_SPEEDUP_THRESHOLD) -> bool:
    """Flag a physically implausible speedup (e.g. >10x on a bandwidth-bound op) — almost
    always elision / timing fraud rather than a real win."""
    try:
        return bool(speedup is not None and speedup > threshold)
    except (TypeError, ValueError):
        return False


# --- Harness integrity (DOMAIN-AGNOSTIC reusable rail) -------------------------------------
# Untrusted candidate code is exec'd in the SAME process as the harness, so it could
# monkey-patch the very functions the oracle uses to judge it (e.g. `torch.allclose = lambda
# *a, **k: True`, or `torch.cuda.synchronize`/`Event`/`time.perf_counter` to fake timing).
# The static checker catches some of this textually; this is the RUNTIME backstop. Snapshot
# the critical callables BEFORE loading the candidate; after loading (and after running it),
# RESTORE any that changed and report them as tamper. The harness then always judges with
# pristine functions, and the attempt is surfaced loudly.
_CRITICAL_CALLABLES = [
    ("torch", "allclose"),
    ("torch", "equal"),
    ("torch", "isnan"),
    ("torch", "isinf"),
    ("torch", "max"),
    ("torch.cuda", "synchronize"),
    ("torch.cuda", "Event"),
    ("torch.cuda", "current_stream"),
    ("torch.cuda", "empty_cache"),
    ("time", "perf_counter"),
]


def _resolve_module(modname: str):
    import importlib

    return importlib.import_module(modname)


def snapshot_harness() -> dict:
    """Capture pristine references to harness-critical callables BEFORE loading untrusted code."""
    snap: dict = {}
    for mod, attr in _CRITICAL_CALLABLES:
        try:
            snap[(mod, attr)] = getattr(_resolve_module(mod), attr)
        except (ImportError, AttributeError):
            pass
    return snap


def restore_harness(snapshot: dict) -> list[str]:
    """Restore any critical callable a candidate monkey-patched; return the list of violations
    (e.g. ['torch.allclose']). Neutralizes the patch (harness reverts to pristine) AND surfaces
    the attempt so it can be flagged as tamper."""
    violations: list[str] = []
    for (mod, attr), pristine in snapshot.items():
        try:
            m = _resolve_module(mod)
            if getattr(m, attr, None) is not pristine:
                violations.append(f"{mod}.{attr}")
                setattr(m, attr, pristine)
        except (ImportError, AttributeError):
            pass
    return violations
