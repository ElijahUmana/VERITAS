"""VERITAS static pre-gate — a thin, opinionated wrapper over KernelBench's
`validate_kernel_static` (vendored, MIT).

This is VERITAS-original code. It *configures and calls* the vendored
`kernel_static_checker.validate_kernel_static` (benchmarks/rmsnorm_lab/vendored/).

Purpose (FLOOR.md §2.2): a pre-execution gate so disguise/bypass cheats die
**before** any GPU spend. Run this CLIENT-SIDE in `kernel_oracle` (no Modal call if it
blocks) and again server-side in the Modal worker as defense-in-depth.

VERITAS policy vs KernelBench defaults:
  - `torch_computation_ops` is promoted from warning -> **ERROR** (block torch.rms_norm,
    torch.matmul, F.*, … — "torch in disguise").
  - backend impl check is forced ERROR (require @triton.jit + tl.* for triton, or
    __global__ + load_inline for cuda).
  - `stream_injection` stays a **WARNING (non-blocking)** on purpose: the stream-bypass cheat
    must reach the GPU so the runtime DUAL TIMER catches it (its named defense). Blocking it
    statically would rob the courtroom of the live dual-timer catch.
"""
from __future__ import annotations

import ast
import os
import sys

# --- Robust import of the vendored checker (works both in-repo and in the flat Modal worker) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_DIRS = [
    os.path.normpath(os.path.join(_HERE, "..", "..", "benchmarks", "rmsnorm_lab", "vendored")),
    _HERE,  # flat layout: kernel_static_checker.py shipped alongside (Modal /root/kb)
]
for _p in _CANDIDATE_DIRS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from kernel_static_checker import validate_kernel_static  # noqa: E402  (vendored, MIT)

# Strict (blocking) checks. `torch_computation_ops` promoted to strict here.
FORBIDDEN = [
    "code_bypass",          # try/except fallback + bare `pass` (inheritance bypass)
    "timing_event_patch",   # monkey-patching cuda.Event/synchronize/perf_counter
    "thread_injection",     # threading/multiprocessing timing games
    "lazy_eval",            # fake/lazy tensor subclasses
    "torch_computation_ops",  # torch.rms_norm / torch.matmul / F.* — "torch in disguise"
]
# Recorded but NON-blocking (stream stays here on purpose — runtime dual-timer is its judge).
WARNINGS = [
    "stream_injection",
    "precision_downgrade",
    "pytorch_wrap",
]


def _strip_docstrings(src: str) -> str:
    """Blank out module/class/function docstrings before static checks.

    The vendored checker scans prose (it strips `#` comments but not docstrings), so a benign
    docstring like "the forward pass" or "decoy code to pass the static gate" would trip the
    bare-word `pass`/`try`/`except` heuristics. We blank ONLY docstring nodes via AST, leaving
    every other string literal intact — crucially the CUDA `cuda_sources = "...__global__..."`
    strings that `check_cuda_impl` must still see. Falls back to the raw source if unparseable.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    lines = src.split("\n")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(first.value.value, str)
            ):
                start = first.lineno
                end = getattr(first, "end_lineno", start) or start
                for ln in range(start, end + 1):
                    if 1 <= ln <= len(lines):
                        lines[ln - 1] = ""
    return "\n".join(lines)


def _has_bare_body_pass(src: str) -> bool:
    """True only for the REAL inheritance-bypass: a function/class whose body is just `pass`
    (after a possible docstring). Incidental `pass` (e.g. `if c: pass`) is not flagged."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return True  # unparseable -> be conservative, keep the vendored flag
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = [
                n for n in node.body
                if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant)
                        and isinstance(n.value.value, str))
            ]
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                return True
    return False


_PASS_MSG = "Contains 'pass' statement (inheritance bypass)"


def static_pregate(src: str, backend: str = "triton", precision: str = "fp32") -> dict:
    """Run the static pre-gate.

    Returns a JSON-able dict:
      {ok: bool, blocked_reason: str|None, errors: [str], warnings: [str], backend, precision}
    `ok=False` => block before GPU spend.
    """
    backend = (backend or "triton").lower()
    code = _strip_docstrings(src)
    valid, errors, warnings = validate_kernel_static(
        code=code,
        backend=backend,            # adds triton_impl / cuda_impl as a strict backend check
        precision=precision,
        forbidden=FORBIDDEN,
        warnings=WARNINGS,
    )
    # AST-precise `pass`: drop the vendored bare-word `pass` error unless a real bare-body pass
    # exists (avoids false-positives on incidental `pass` in honest code).
    if _PASS_MSG in errors and not _has_bare_body_pass(src):
        errors = [e for e in errors if e != _PASS_MSG]
        valid = len(errors) == 0

    return {
        "ok": bool(valid),
        "blocked_reason": (None if valid else "; ".join(errors) or "static check failed"),
        "errors": list(errors),
        "warnings": list(warnings),
        "backend": backend,
        "precision": precision,
    }


if __name__ == "__main__":  # tiny self-check (no torch needed)
    honest = "import triton\nimport triton.language as tl\n@triton.jit\ndef k():\n    x = tl.load(0)\n"
    disguise = "import torch\ndef f(x):\n    return torch.rms_norm(x, [64])\n"
    print("honest   ->", static_pregate(honest))
    print("disguise ->", static_pregate(disguise))
