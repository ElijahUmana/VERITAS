"""VERITAS demo + verification harness (owned by demo-verifier, Task #5).

Stdlib-only support code shared by ``crucible/demo.py`` (the one-command demo
runner) and ``tests/adversarial_selftest.py`` (the adversarial self-test).

Modules:
  workshop  — zero-dependency HTTP client + readback assertions for the local
              Raindrop Workshop daemon (:5899). The "courtroom" verifier.
  fallback  — record/replay cached-evidence layer so a WiFi/Modal hiccup can
              never kill the demo (FLOOR.md §1 "rehearsed inputs, cached fallbacks").
  beats     — beat timing, the <60s budget tracker, and colored narration.

Nothing here imports the ``crucible`` engine package at import time; engine
modules are imported lazily inside the runner/test so the harness stays usable
even while the spine is still being built.
"""

from __future__ import annotations

__all__ = ["workshop", "fallback", "beats"]
