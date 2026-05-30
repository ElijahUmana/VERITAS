"""Beat timing, the <60s budget tracker, and colored narration.

The demo is a sequence of timed "beats" (FLOOR.md §1). This module gives the
runner:
  • a Timeline that records each beat's wall-clock and renders a final report
    that PASSES/FAILS on the <60s target,
  • a Beat context manager that times a block and flags per-beat budget overruns,
  • narration helpers (GREEN/RED verdict lines) matching the phase-zero style.

Pure stdlib. Honors NO_COLOR.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t: str) -> str: return _c("32", t)
def red(t: str) -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def cyan(t: str) -> str: return _c("36", t)
def bold(t: str) -> str: return _c("1", t)
def dim(t: str) -> str: return _c("2", t)

# Short aliases (used by crucible/demo.py).
grn = green
ylw = yellow
cyn = cyan


TARGET_SECONDS = 60.0


def say(msg: str) -> None:
    print(msg, flush=True)


def narrate(msg: str) -> None:
    """Stage narration line (FLOOR.md §1 'Narration:' cues)."""
    print(dim("  » ") + cyan(msg), flush=True)


def verdict_green(msg: str) -> None:
    print("  " + green("● GREEN ") + msg, flush=True)


def verdict_red(msg: str) -> None:
    print("  " + red("● RED   ") + msg, flush=True)


def badge(source: str) -> str:
    """LIVE / CACHED badge for a beat's data source."""
    if source == "live":
        return green("[LIVE]")
    if source == "cached":
        return yellow("[CACHED]")
    return dim(f"[{source}]")


@dataclass
class BeatRecord:
    title: str
    budget_s: float | None
    elapsed_s: float
    ok: bool

    @property
    def over_budget(self) -> bool:
        return self.budget_s is not None and self.elapsed_s > self.budget_s


@dataclass
class Timeline:
    target_s: float = TARGET_SECONDS
    records: list[BeatRecord] = field(default_factory=list)
    _start: float = field(default_factory=time.monotonic)

    def reset(self) -> None:
        self._start = time.monotonic()
        self.records.clear()

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

    @contextmanager
    def beat(self, title: str, budget_s: float | None = None):
        bar = "─" * 64
        print("\n" + bold(cyan(bar)), flush=True)
        b = "" if budget_s is None else dim(f"  (budget {budget_s:.0f}s)")
        print(bold(cyan("▶ " + title)) + b, flush=True)
        print(cyan(bar), flush=True)
        t0 = time.monotonic()
        ok = True
        try:
            yield self
        except Exception:
            ok = False
            raise
        finally:
            dt = time.monotonic() - t0
            rec = BeatRecord(title, budget_s, dt, ok)
            self.records.append(rec)
            tag = green(f"{dt:5.1f}s")
            if rec.over_budget:
                tag = yellow(f"{dt:5.1f}s  ⚠ over {budget_s:.0f}s budget")
            print(dim(f"  ⤷ beat time {tag}  ·  total {self.elapsed_s:5.1f}s"), flush=True)

    def report(self) -> bool:
        total = self.elapsed_s
        within = total <= self.target_s
        bar = "═" * 64
        print("\n" + bold(bar))
        print(bold("  TIMING REPORT"))
        print(bar)
        for r in self.records:
            mark = green("ok ") if r.ok else red("ERR")
            over = yellow("  ⚠ over budget") if r.over_budget else ""
            budget = "" if r.budget_s is None else dim(f" / {r.budget_s:.0f}s")
            print(f"  {mark}  {r.elapsed_s:5.1f}s{budget}  {r.title}{over}")
        verdict = green(f"{total:.1f}s  ≤ {self.target_s:.0f}s  ✓ WITHIN BUDGET") if within \
            else red(f"{total:.1f}s  > {self.target_s:.0f}s  ✗ OVER BUDGET")
        print(bar)
        print(f"  TOTAL: {verdict}")
        print(bar, flush=True)
        return within


def banner(title: str, subtitle: str = "") -> None:
    bar = "═" * 64
    print(bold(cyan(bar)))
    print(bold(cyan("  " + title)))
    if subtitle:
        print(dim("  " + subtitle))
    print(bold(cyan(bar)), flush=True)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _selfcheck() -> int:
    tl = Timeline(target_s=60.0)
    banner("BEATS SELF-CHECK", "timing + narration primitives")
    with tl.beat("Beat A", budget_s=5):
        narrate("an agent just lied")
        verdict_red("fabricated case 404 — does not exist")
        time.sleep(0.05)
    with tl.beat("Beat B", budget_s=0.0):  # force an over-budget warning
        verdict_green("real case 200 — confirmed")
        time.sleep(0.02)
    within = tl.report()
    print(green("\nbeats OK") if within and len(tl.records) == 2 else red("\nbeats FAILED"))
    return 0 if (within and len(tl.records) == 2) else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
