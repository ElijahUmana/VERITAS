"""VERITAS Modal verifier — the external mechanical oracle (deploy-once, call-many).

    modal deploy modal/verifier_app.py        # ONCE — builds the T4 image, registers the app
    # then call modal.Function.from_name("veritas-verifier", "verify_candidate").remote(payload)
    # NEVER `modal run` per candidate.

An isolated T4 GPU sandbox compiles + runs ONE candidate against the RMSNorm reference and
returns a JSON-only verdict. It NEVER trusts candidate-reported numbers — every number is
measured here, by the oracle. Hardening (FLOOR.md §2.2):

  - correctness: 5 trials (seeds from 42) + HIDDEN tests (extra seed, extra shape, rand_mix);
    candidate runs on a CLONE, reference recomputed from a pristine copy; allocator NaN-poison +
    output materialization (isnan/isinf/shape/dtype); torch.allclose at fp32 atol=rtol=1e-2.
  - speed: 5 warmup -> 100 trials cuda_event with L2 clears (vendored KernelBench timing).
  - anti-tamper: DUAL timer (cuda_event vs triton do_bench) reject on >1.5x disagreement;
    >10x excessive-speedup reject; static pre-gate re-checked server-side (defense in depth).

A candidate that errors is REFUTED (loud), never a false pass. A *harness* error returns
verifier_status=ERROR / verdict=unverified so the promotion gate can never accept it.

VERITAS-original code. It imports the vendored KernelBench timing + static-checker (MIT) and the
VERITAS anti-tamper module, all shipped into the image under /root/kb. The three small model
loaders below are adapted from KernelBench src/kernelbench/eval.py (MIT) — attributed inline.
"""
from __future__ import annotations

import pathlib

import modal

APP_NAME = "veritas-verifier"
GPU = "T4"  # cheapest real GPU; the demo substrate. Override at deploy via env if desired.

_HERE = pathlib.Path(__file__).resolve().parent          # VERITAS/modal
_ROOT = _HERE.parent                                       # VERITAS
_VENDORED = _ROOT / "benchmarks" / "rmsnorm_lab" / "vendored"
_ORACLE = _ROOT / "crucible" / "oracle"

# Image: torch (bundles a matching triton on linux) + numpy. nvcc is NOT required — every
# candidate is Triton/torch. The proven phase-zero combo (debian_slim 3.12 + pip torch) gave
# torch 2.12+cu130 live on T4. We ship the vendored timing + static-checker + anti-tamper flat
# under /root/kb and import them in the worker.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
    .add_local_file(_VENDORED / "timing.py", "/root/kb/timing.py", copy=True)
    .add_local_file(_VENDORED / "kernel_static_checker.py", "/root/kb/kernel_static_checker.py", copy=True)
    .add_local_file(_ORACLE / "static_checker.py", "/root/kb/static_checker.py", copy=True)
    .add_local_file(_ORACLE / "anti_tamper.py", "/root/kb/anti_tamper.py", copy=True)
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Model loaders — adapted from KernelBench src/kernelbench/eval.py (MIT).
# ---------------------------------------------------------------------------
def _set_seed(seed: int):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _load_reference(src: str):
    """exec the reference source -> (Model, get_init_inputs, get_inputs)."""
    ctx: dict = {}
    compile(src, "<reference>", "exec")
    exec(src, ctx)  # noqa: S102 — trusted, fixed reference source
    return ctx["Model"], ctx["get_init_inputs"], ctx["get_inputs"]


def _load_candidate(src: str, entry_point: str = "ModelNew"):
    """Load ModelNew via a tempfile import (required for @triton.jit; also works for plain torch).
    Returns (ModelNew, tempfile_path). Adapted from KernelBench load_custom_model_with_tempfile."""
    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    spec = importlib.util.spec_from_file_location("veritas_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # candidate exceptions surface here -> REFUTED
    return getattr(module, entry_point), path


class _CandidateError(Exception):
    """Candidate-side failure (compile/import/forward). -> REFUTED, verifier_status=OK."""


# ---------------------------------------------------------------------------
# One correctness trial with full anti-tamper hardening.
# ---------------------------------------------------------------------------
def _run_trial(ref_model, cand_model, inputs, tol, device, anti_tamper, snapshot):
    import torch

    pristine = anti_tamper.clone_tensors(inputs)

    # Reference from an independent clone of the pristine inputs.
    ref_out = ref_model(*anti_tamper.clone_tensors(pristine))
    torch.cuda.synchronize(device)

    # Poison the allocator's free pool with NaN blocks of the output shape (kills result-reuse).
    anti_tamper.poison_free_pool(ref_out.shape, ref_out.dtype, device)

    # Candidate runs on its OWN clones.
    cand_inputs = anti_tamper.clone_tensors(pristine)
    try:
        cand_out = cand_model(*cand_inputs)
    except Exception as e:  # candidate forward blew up -> trial fails (candidate error, short-circuit)
        anti_tamper.restore_harness(snapshot)
        return {"passed": False, "materialized": False, "mutated": False, "patched": [],
                "candidate_error": True,
                "reason": f"candidate raised in forward: {type(e).__name__}: {e}", "max_abs_err": None}
    # Revert any harness monkey-patch the candidate forward installed BEFORE we trust any torch fn.
    patched = anti_tamper.restore_harness(snapshot)
    try:
        torch.cuda.synchronize(device)
    except Exception as e:
        return {"passed": False, "materialized": False, "mutated": False, "patched": patched,
                "candidate_error": True,
                "reason": f"candidate caused a CUDA error: {type(e).__name__}: {e}", "max_abs_err": None}

    mutated, mut_detail = anti_tamper.detect_input_mutation(cand_inputs, pristine)
    mat_ok, mat_reason = anti_tamper.assert_materialized(cand_out, ref_out.shape, ref_out.dtype)
    if not mat_ok:
        return {"passed": False, "materialized": False, "mutated": bool(mutated), "patched": patched,
                "candidate_error": False,
                "reason": f"output not materialized: {mat_reason}", "max_abs_err": None}

    close = bool(torch.allclose(ref_out, cand_out, atol=tol, rtol=tol))
    max_err = float((ref_out - cand_out).abs().max().item())
    reason = ""
    if patched:
        reason = f"harness integrity violation: candidate patched {patched}"
    elif not close:
        reason = f"allclose failed: max_abs_err={max_err:.6g} > tol={tol:g}"
        if mutated:
            reason += f" (input mutated: {mut_detail})"
    return {"passed": (close and not patched), "materialized": True, "mutated": bool(mutated),
            "patched": patched, "candidate_error": False, "max_abs_err": max_err, "reason": reason}


def _hidden_trials(ref_model, cand_model, tol, device, dtype, features, anti_tamper, snapshot):
    """Hidden tests the candidate never sees: extra seed + extra shape + rand_mix distribution."""
    import torch

    results = {}
    # (a) extra shape, hidden seed, uniform
    _set_seed(987654321)
    x_a = torch.rand(8, features, 192, 320, device=device, dtype=dtype)
    results["hidden_extra_shape"] = _run_trial(ref_model, cand_model, [x_a], tol, device, anti_tamper, snapshot)
    if results["hidden_extra_shape"].get("candidate_error"):
        return results
    # (b) rand_mix: normal + sparse heavy outliers, different shape
    _set_seed(192837465)
    base = torch.randn(12, features, 224, 224, device=device, dtype=dtype)
    outliers = (torch.rand(12, features, 224, 224, device=device, dtype=dtype) > 0.99).to(dtype) * 50.0
    results["hidden_rand_mix"] = _run_trial(ref_model, cand_model, [base + outliers], tol, device, anti_tamper, snapshot)
    return results


# ---------------------------------------------------------------------------
# The deployed verifier function.
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, timeout=900, retries=0, block_network=True)
def verify_candidate(payload: dict) -> dict:
    """Run ONE candidate through the hardened oracle. Returns a JSON-only verdict dict.

    payload keys (all optional except the two sources):
      reference_src, candidate_src, claim_id, candidate_id, backend ("triton"),
      precision ("fp32"), tolerance (1e-2), num_correct_trials (5), num_warmup (5),
      num_perf_trials (100), seed (42), dual_timer_threshold (1.5),
      excessive_speedup_threshold (10.0), run_static (True)
    """
    import sys
    import traceback

    sys.path.insert(0, "/root/kb")
    import torch  # noqa: E402

    claim_id = str(payload.get("claim_id", "claim"))
    candidate_id = str(payload.get("candidate_id", "candidate"))
    backend = str(payload.get("backend", "triton"))
    precision = str(payload.get("precision", "fp32"))
    tol = float(payload.get("tolerance", 1e-2))
    n_correct = int(payload.get("num_correct_trials", 5))
    n_warmup = int(payload.get("num_warmup", 5))
    n_perf = int(payload.get("num_perf_trials", 100))
    seed = int(payload.get("seed", 42))
    dual_thr = float(payload.get("dual_timer_threshold", 1.5))
    exc_thr = float(payload.get("excessive_speedup_threshold", 10.0))
    run_static = bool(payload.get("run_static", True))

    verdict = {
        "verdict": "unverified", "claim_id": claim_id, "candidate_id": candidate_id,
        "oracle_type": "kernel", "correctness_passed": False, "speedup": None,
        "tamper_detected": False, "verifier_status": "OK", "blocked_reason": None,
        "hardware": None, "measured_by": "modal-oracle", "details": {}, "error": None,
    }

    try:
        import anti_tamper
        import timing
        import static_checker

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in the Modal worker (harness/infra error)")
        device = torch.device("cuda", torch.cuda.current_device())
        verdict["hardware"] = torch.cuda.get_device_name(device)
        # Sandbox identity — lets a fan-out prove N candidates ran on M distinct Modal containers
        # (the megastructure made real). Modal sets MODAL_TASK_ID per container.
        import os as _os
        import platform as _platform
        verdict["details"]["sandbox"] = {
            "modal_task_id": _os.environ.get("MODAL_TASK_ID"),
            "node": _platform.node(),
        }
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

        candidate_src = payload["candidate_src"]
        reference_src = payload["reference_src"]

        # ---- Static pre-gate (server-side re-check; client gate already ran for free) ----
        static = static_checker.static_pregate(candidate_src, backend=backend, precision=precision)
        verdict["details"]["static"] = static
        if run_static and not static["ok"]:
            verdict.update({
                "verdict": "blocked", "tamper_detected": True,
                "blocked_reason": f"static pre-gate: {static['blocked_reason']}",
            })
            return _json_safe(verdict)

        # ---- Load reference (harness-owned; failure => ERROR) ----
        try:
            Model, get_init_inputs, get_inputs = _load_reference(reference_src)
        except Exception as e:
            raise RuntimeError(f"reference source failed to load (harness error): {e}")

        _set_seed(seed)
        init_inputs = [
            x.to(device=device, dtype=dtype) if isinstance(x, torch.Tensor) else x
            for x in get_init_inputs()
        ]
        with torch.no_grad():
            _set_seed(seed)
            ref_model = Model(*init_inputs).to(device=device, dtype=dtype)
        features = int(init_inputs[0]) if isinstance(init_inputs[0], int) else 64

        # ---- Snapshot harness-critical callables BEFORE loading untrusted candidate code ----
        harness_snapshot = anti_tamper.snapshot_harness()

        # ---- Load candidate (failure => REFUTED, candidate's fault) ----
        cand_path = None
        try:
            ModelNew, cand_path = _load_candidate(candidate_src)
            with torch.no_grad():
                _set_seed(seed)
                cand_model = ModelNew(*init_inputs).to(device=device, dtype=dtype)
            torch.cuda.synchronize(device)
        except Exception as e:
            anti_tamper.restore_harness(harness_snapshot)
            verdict.update({
                "verdict": "refuted",
                "blocked_reason": f"candidate failed to compile/load: {type(e).__name__}: {e}",
            })
            verdict["details"]["load_error"] = traceback.format_exc()
            return _json_safe(verdict)

        # Import-time harness integrity: did the candidate monkey-patch a judging function?
        import_patches = anti_tamper.restore_harness(harness_snapshot)
        if import_patches:
            verdict.update({
                "verdict": "refuted", "tamper_detected": True,
                "blocked_reason": f"harness integrity: candidate patched {import_patches} at import",
            })
            verdict["details"]["harness_integrity"] = {"import_patches": import_patches}
            return _json_safe(verdict)

        # ---- Correctness: 5 seeded trials + hidden tests (all anti-tamper hardened) ----
        trial_seeds = _derive_seeds(seed, n_correct)
        trials = {}
        with torch.no_grad():
            for i, ts in enumerate(trial_seeds):
                _set_seed(ts)
                inputs = [
                    x.to(device=device, dtype=dtype) if isinstance(x, torch.Tensor) else x
                    for x in get_inputs()
                ]
                trials[f"seed_{i}"] = _run_trial(ref_model, cand_model, inputs, tol, device, anti_tamper, harness_snapshot)
                if trials[f"seed_{i}"].get("candidate_error"):
                    break  # crashed candidate (incl. CUDA illegal access) -> stop; don't run on a poisoned context
            else:
                hidden = _hidden_trials(ref_model, cand_model, tol, device, dtype, features, anti_tamper, harness_snapshot)
                trials.update(hidden)
        verdict["details"]["correctness"] = trials

        # A candidate that crashed is REFUTED (its fault) — NOT a harness ERROR, and never a false pass.
        crashed = [t for t in trials.values() if t.get("candidate_error")]
        if crashed:
            verdict.update({
                "verdict": "refuted", "tamper_detected": False,
                "blocked_reason": f"candidate crashed during execution: {crashed[0]['reason']}",
            })
            return _json_safe(verdict)

        any_non_materialized = any(not t["materialized"] for t in trials.values())
        any_mutated = any(t["mutated"] for t in trials.values())
        any_patched = any(t.get("patched") for t in trials.values())
        correctness_passed = all(t["passed"] for t in trials.values())
        verdict["correctness_passed"] = bool(correctness_passed)
        errs = [t["max_abs_err"] for t in trials.values() if t["max_abs_err"] is not None]
        worst_err = max(errs) if errs else float("nan")

        if not correctness_passed:
            tamper = bool(any_non_materialized or any_mutated or any_patched)
            if any_patched:
                reason = "harness integrity violation: candidate patched a judging function at runtime"
            elif any_non_materialized:
                reason = "non-materialized output (uninitialized / reused buffer caught by poison+isnan)"
            elif any_mutated:
                reason = "input mutation: candidate altered inputs; reference recomputed from pristine copy -> mismatch"
            else:
                reason = f"correctness failed (worst max_abs_err={worst_err:.6g} > tol={tol:g})"
            verdict.update({"verdict": "refuted", "tamper_detected": tamper, "blocked_reason": reason})
            return _json_safe(verdict)

        # ---- Speed + DUAL timer (only reached if correctness fully passed) ----
        _set_seed(seed)
        perf_inputs = [
            x.to(device=device, dtype=dtype) if isinstance(x, torch.Tensor) else x
            for x in get_inputs()
        ]
        cand_cuda = _mean(timing.time_execution_with_cuda_event(
            cand_model, perf_inputs, num_warmup=n_warmup, num_trials=n_perf, verbose=False, device=device))
        ref_cuda = _mean(timing.time_execution_with_cuda_event(
            ref_model, perf_inputs, num_warmup=n_warmup, num_trials=n_perf, verbose=False, device=device))
        cand_db = _mean(timing.time_execution_with_do_bench_interface(
            cand_model, perf_inputs, verbose=False, device=device))
        ref_db = _mean(timing.time_execution_with_do_bench_interface(
            ref_model, perf_inputs, verbose=False, device=device))

        speedup_cuda = (ref_cuda / cand_cuda) if cand_cuda > 0 else float("inf")
        speedup_db = (ref_db / cand_db) if cand_db > 0 else float("inf")

        disagree, ratio = anti_tamper.dual_timer_disagreement(cand_cuda, cand_db, dual_thr)
        excessive = anti_tamper.excessive_speedup(speedup_cuda, exc_thr) or anti_tamper.excessive_speedup(speedup_db, exc_thr)

        verdict["details"]["speed"] = {
            "cand_cuda_event_ms": cand_cuda, "ref_cuda_event_ms": ref_cuda,
            "cand_do_bench_ms": cand_db, "ref_do_bench_ms": ref_db,
            "speedup_cuda_event": speedup_cuda, "speedup_do_bench": speedup_db,
            "num_warmup": n_warmup, "num_perf_trials": n_perf,
        }
        verdict["details"]["anti_tamper"] = {
            "dual_timer_ratio": ratio, "dual_timer_threshold": dual_thr, "dual_timer_disagree": bool(disagree),
            "excessive_speedup": bool(excessive), "excessive_threshold": exc_thr,
            "materialized_all": True, "input_mutation": False,
        }

        if disagree or excessive:
            why = []
            if disagree:
                why.append(f"dual-timer disagreement {ratio:.2f}x > {dual_thr:g}x (work hidden on non-default stream)")
            if excessive:
                why.append(f"excessive speedup (cuda_event {speedup_cuda:.2f}x / do_bench {speedup_db:.2f}x > {exc_thr:g}x)")
            verdict.update({
                "verdict": "refuted", "tamper_detected": True,
                "speedup": _finite_or_none(speedup_db), "blocked_reason": "; ".join(why),
            })
            return _json_safe(verdict)

        # ---- CONFIRMED ----
        verdict.update({
            "verdict": "confirmed", "tamper_detected": False,
            "speedup": _finite_or_none(speedup_db),  # trustworthy full-sync timer
        })
        return _json_safe(verdict)

    except Exception as e:  # HARNESS error only — candidate errors handled above
        verdict.update({
            "verdict": "unverified", "verifier_status": "ERROR",
            "error": f"{type(e).__name__}: {e}",
        })
        verdict["details"]["harness_traceback"] = traceback.format_exc()
        return _json_safe(verdict)


@app.function(image=image, gpu=GPU, timeout=300, retries=0)
def selftest() -> dict:
    """Prove the image: torch+CUDA on T4 and a real Triton kernel JIT-compiles and runs."""
    import sys
    sys.path.insert(0, "/root/kb")
    import torch
    out = {"torch": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
    try:
        import triton
        import triton.language as tl
        out["triton"] = str(triton.__version__)

        @triton.jit
        def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

        n = 4096
        x = torch.rand(n, device="cuda")
        y = torch.rand(n, device="cuda")
        o = torch.empty_like(x)
        _add[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
        torch.cuda.synchronize()
        out["triton_kernel_ok"] = bool(torch.allclose(o, x + y))
    except Exception as e:
        out["triton_error"] = f"{type(e).__name__}: {e}"
        out["triton_kernel_ok"] = False
    return out


# ---------------------------------------------------------------------------
# Small pure helpers (module-level so they ship with the app).
# ---------------------------------------------------------------------------
def _derive_seeds(base: int, n: int) -> list[int]:
    import torch
    torch.manual_seed(base)
    return [int(torch.randint(0, 2**31 - 1, (1,)).item()) for _ in range(n)]


def _mean(times) -> float:
    vals = [float(t) for t in times] if hasattr(times, "__iter__") else [float(times)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _finite_or_none(x):
    try:
        x = float(x)
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def _json_safe(obj):
    """Recursively coerce to JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, bool, int)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else str(obj)
    return str(obj)


@app.local_entrypoint()
def main():
    """Dev smoke (modal run): selftest + verify the honest candidate. Production uses
    `modal deploy` + Function.from_name (see modal/run_oracle_proof.py)."""
    import json

    st = selftest.remote()
    print("[selftest]", json.dumps(st, indent=2))

    ref_src = (_ROOT / "benchmarks/rmsnorm_lab/reference.py").read_text()
    # Build a self-contained reference source via the module's emitter:
    import importlib.util
    spec = importlib.util.spec_from_file_location("veritas_reference", _ROOT / "benchmarks/rmsnorm_lab/reference.py")
    refmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refmod)
    reference_src = refmod.reference_source()

    cand_src = (_ROOT / "benchmarks/rmsnorm_lab/candidates/good_rehearsed.py").read_text()
    res = verify_candidate.remote({
        "reference_src": reference_src, "candidate_src": cand_src,
        "claim_id": "demo", "candidate_id": "good_rehearsed",
    })
    print("[verify good_rehearsed]", json.dumps(res, indent=2))
