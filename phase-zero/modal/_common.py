"""Shared helpers for the Modal Phase-Zero live-verification suite.

These helpers verify live behavior, surface errors clearly, and avoid masking
exceptions.

GPU choice: defaults to the cheapest real GPU (T4, ~$0.000164/s) so we prove the
*mechanism* without burning credits. The actual demo scales to H100/B200 — that is a
deliberate, stated choice (test small, demo big), NOT a scope reduction.
Override with:  MODAL_VERIFY_GPU=H100 python 01_sandbox_gpu.py
"""
from __future__ import annotations
import os
import sys
import time
import pathlib
import traceback
from datetime import datetime, timezone

# --- Load .env from repo root so MODAL_* / OPENAI_* are available --------------
# Modal `modal run` workers import a copied `/root/_common.py`; in that layout there
# is no repo grandparent. Local execution gets the real repo root, remote execution
# simply skips dotenv and uses environment/defaults.
_HERE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else pathlib.Path.cwd()
try:
    from dotenv import load_dotenv
    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except Exception:  # dotenv missing is non-fatal; env may already be exported
    pass

APP_NAME = os.environ.get("MODAL_VERIFY_APP", "veritas-verify")
GPU = os.environ.get("MODAL_VERIFY_GPU", "T4")          # cheapest real GPU by default
GPU_IS_CPU = GPU.lower() in ("", "cpu", "none")
VERIFY_VOLUME = os.environ.get("MODAL_VERIFY_VOLUME", "veritas-verify-vol")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"[{_ts()}] \033[32mPASS\033[0m  {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[{_ts()}] \033[31mFAIL\033[0m  {msg}", flush=True)


def require(cond: bool, msg: str) -> None:
    """Hard assertion that prints a clear PASS/FAIL line. Raises on failure."""
    if cond:
        ok(msg)
    else:
        fail(msg)
        raise AssertionError(msg)


def banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}", flush=True)


def run(main_fn) -> None:
    """Execute a verification `main_fn`, timing it, surfacing every error.

    Exit code 0 == verified live. Non-zero == real failure (auth, quota, bug).
    """
    name = pathlib.Path(sys.argv[0]).name
    banner(f"MODAL PHASE-ZERO :: {name} :: app={APP_NAME} gpu={GPU}")
    t0 = time.time()
    try:
        main_fn()
    except Exception as exc:  # noqa: BLE001 — we re-raise after reporting; nothing swallowed
        dt = time.time() - t0
        fail(f"{name} after {dt:.1f}s :: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
    dt = time.time() - t0
    ok(f"{name} COMPLETE in {dt:.1f}s — verified live")
