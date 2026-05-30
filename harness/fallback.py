"""Record/replay cached-evidence layer — the backup flow (FLOOR.md §1).

Goal: a WiFi or Modal hiccup can NEVER kill the demo. Every live beat
(legal citation lookup, Modal kernel verification, OpenAI generation) is wrapped
so that:

  • In ``auto`` mode (default): try the live function; on ANY failure fall back to
    a frozen cached artifact. If live succeeds and recording is enabled, refresh
    the cache so the rehearsed artifact stays current.
  • In ``cached`` mode: never touch the network — replay the frozen artifact.
    This is the guaranteed-deterministic floor used for the rehearsed <60s run.
  • In ``live`` mode: live only, no fallback (used when recording or when a test
    explicitly wants to exercise the real path).
  • In ``record`` mode: run live and overwrite the cache, then return live.

Each call returns an ``Outcome(value, source, error)`` where ``source`` is
``"live"`` or ``"cached"`` so the demo can render a LIVE vs CACHED badge and the
self-test can assert which path ran.

Cached artifacts are plain JSON on disk, human-readable and git-diffable, so the
"rehearsed inputs" are auditable.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Callable

# Demo execution mode resolved from the environment (CLI sets this).
#   auto   — prefer live, fall back to cache (resilient default)
#   cached — frozen artifacts only, zero network (guaranteed floor)
#   live   — live only, no fallback
#   record — live + overwrite cache
VALID_MODES = ("auto", "cached", "live", "record")


def resolve_mode(explicit: str | None = None) -> str:
    mode = (explicit or os.environ.get("VERITAS_DEMO_MODE") or "auto").strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"VERITAS_DEMO_MODE must be one of {VALID_MODES}, got {mode!r}")
    return mode


@dataclass
class Outcome:
    value: Any
    source: str            # "live" | "cached"
    error: str | None = None  # populated when live failed and we fell back

    @property
    def is_cached(self) -> bool:
        return self.source == "cached"

    @property
    def is_live(self) -> bool:
        return self.source == "live"


class CacheMiss(RuntimeError):
    pass


class CacheStore:
    """A directory of named JSON artifacts."""

    def __init__(self, base_dir: str | os.PathLike):
        self.base = pathlib.Path(base_dir)

    def path(self, key: str) -> pathlib.Path:
        safe = key if key.endswith(".json") else f"{key}.json"
        return self.base / safe

    def has(self, key: str) -> bool:
        return self.path(key).is_file()

    def load(self, key: str) -> Any:
        p = self.path(key)
        if not p.is_file():
            raise CacheMiss(f"no cached artifact at {p}")
        return json.loads(p.read_text())

    def save(self, key: str, value: Any) -> pathlib.Path:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return p

    def with_fallback(self, key: str, live_fn: Callable[[], Any], *,
                      mode: str | None = None) -> Outcome:
        """Run live_fn / replay cache per the resolved mode. Never raises for a
        cache-backed beat unless BOTH live and cache are unavailable."""
        mode = resolve_mode(mode)

        if mode == "cached":
            return Outcome(self.load(key), "cached")

        if mode in ("live", "record", "auto"):
            try:
                value = live_fn()
            except Exception as exc:  # noqa: BLE001 — we deliberately convert to fallback
                if mode == "live":
                    raise
                # auto / record: fall back to the frozen artifact
                if self.has(key):
                    return Outcome(self.load(key), "cached", error=_short(exc))
                raise CacheMiss(
                    f"live failed ({_short(exc)}) and no cached artifact at {self.path(key)}"
                ) from exc
            # live succeeded
            if mode in ("record", "auto") and _should_record(mode):
                try:
                    self.save(key, value)
                except Exception:
                    pass  # caching is best-effort; never fail the demo on a cache write
            return Outcome(value, "live")

        raise AssertionError(f"unreachable mode {mode!r}")


def _should_record(mode: str) -> bool:
    # In 'record' always persist. In 'auto', only refresh the cache if explicitly
    # asked (VERITAS_DEMO_REFRESH=1) so a normal live run doesn't mutate rehearsed
    # artifacts under the demo-runner's feet.
    if mode == "record":
        return True
    return os.environ.get("VERITAS_DEMO_REFRESH", "").strip() in ("1", "true", "yes")


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _selfcheck() -> int:
    import tempfile

    ok_all = True

    def chk(label, cond, extra=""):
        nonlocal ok_all
        mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
        ok_all = ok_all and bool(cond)
        print(f"  {mark}  {label}{(' — ' + extra) if extra else ''}")

    with tempfile.TemporaryDirectory() as d:
        store = CacheStore(d)

        # record: live runs and persists
        out = store.with_fallback("beat", lambda: {"v": 1, "src": "live"}, mode="record")
        chk("record returns live", out.is_live and out.value["v"] == 1)
        chk("record persisted cache", store.has("beat"))

        # cached: replays without calling live
        out = store.with_fallback("beat", lambda: (_ for _ in ()).throw(RuntimeError("should not run")),
                                  mode="cached")
        chk("cached replays artifact", out.is_cached and out.value["v"] == 1)

        # auto + live failure: falls back to cache
        out = store.with_fallback("beat", lambda: (_ for _ in ()).throw(RuntimeError("net down")),
                                  mode="auto")
        chk("auto falls back on failure", out.is_cached and out.error and "net down" in out.error)

        # auto + live success: returns live
        out = store.with_fallback("beat", lambda: {"v": 2, "src": "live"}, mode="auto")
        chk("auto prefers live on success", out.is_live and out.value["v"] == 2)

        # live mode: failure raises (no silent swallow)
        try:
            store.with_fallback("beat", lambda: (_ for _ in ()).throw(RuntimeError("boom")), mode="live")
            chk("live mode raises on failure", False)
        except RuntimeError:
            chk("live mode raises on failure", True)

        # cache miss with no artifact: raises CacheMiss (fails loud)
        try:
            store.with_fallback("never", lambda: (_ for _ in ()).throw(RuntimeError("x")), mode="auto")
            chk("missing artifact fails loud", False)
        except CacheMiss:
            chk("missing artifact fails loud", True)

    print("\033[32mfallback layer OK\033[0m" if ok_all else "\033[31mfallback layer FAILED\033[0m")
    return 0 if ok_all else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
