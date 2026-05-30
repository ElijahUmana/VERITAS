"""crucible/raindrop_bridge.py — the native Raindrop trace bridge (FLOOR §4).

Wires the OpenAI Agents SDK's tracing into Raindrop Workshop so EVERY
agent / LLM / tool span the generator (and the cold-open agent) produce reaches
the courtroom. This is the path proven live in phase-zero battery #12
(``openai-VERIFIED.md``): ``create_raindrop_openai_agents(...)`` installs a
``RaindropTracingProcessor`` via the SDK's native ``add_trace_processor`` hook
(the wrapper auto-registers it, with double-wrap protection), and additional
processors fan out *additively* through the same hook.

This complements — does not duplicate — ``crucible/trace.py``: that module emits
the explicit ``crucible.*`` OTLP domain spans (mission/claim/oracle/ledger);
THIS module ensures the model's own agent/LLM/tool spans are captured, and lets
a run be wrapped in a ``trace()`` carrying ``crucible.*`` metadata so the SDK
trace and the domain trace share context in Workshop.

Design choices:
  * No key -> graceful no-op (returns ``None``), so the generator still runs and
    the cold open still lands; the missing-key reason is printed LOUDLY.
  * The native ``add_trace_processor`` hook is exercised EXPLICITLY (not only via
    the wrapper) by attaching a :class:`CrucibleSpanCounter`, which both proves
    the fan-out fired and gives a verifiable span count for the demo self-test.
"""
from __future__ import annotations

import contextlib
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from agents import custom_span, trace
from agents.tracing import TracingProcessor, add_trace_processor


def _log(msg: str) -> None:
    print(f"[raindrop_bridge] {msg}", file=sys.stderr)


class CrucibleSpanCounter(TracingProcessor):
    """A tiny additive processor (registered through the native ``add_trace_processor``
    hook) that counts the agent/LLM/tool spans flowing to Workshop. It proves the
    fan-out fired and gives the demo a verifiable signal (battery #12 pattern)."""

    def __init__(self) -> None:
        self.traces = 0
        self.spans = 0
        self.span_kinds: list[str] = []

    def on_trace_start(self, trace_obj: Any) -> None:  # noqa: D401
        self.traces += 1

    def on_trace_end(self, trace_obj: Any) -> None:
        pass

    def on_span_start(self, span: Any) -> None:
        pass

    def on_span_end(self, span: Any) -> None:
        self.spans += 1
        try:
            self.span_kinds.append(type(span.span_data).__name__)
        except Exception:
            self.span_kinds.append("<unknown>")

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


@dataclass
class BridgeHandle:
    """Handle returned by :func:`install_raindrop_bridge`. ``flush()`` pushes any
    buffered spans to Workshop; ``counter`` exposes the captured span count."""

    client: Any = None
    counter: Optional[CrucibleSpanCounter] = None
    enabled: bool = False
    extra: list = field(default_factory=list)

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                _log("flush failed (non-fatal):\n" + traceback.format_exc())

    @property
    def span_count(self) -> int:
        return self.counter.spans if self.counter else 0


def install_raindrop_bridge(
    *,
    user_id: str,
    convo_id: str = "autoresearch-hackathon",
    extra_processors: Optional[list] = None,
    count_spans: bool = True,
    debug: bool = False,
) -> Optional[BridgeHandle]:
    """Install the Raindrop bridge for the current process.

    Returns a :class:`BridgeHandle` (call ``.flush()`` after your run), or
    ``None`` if ``RAINDROP_WRITE_KEY`` is absent or the bridge cannot be created.
    Never raises — a tracing failure must not break the agent run; the reason is
    logged loudly instead.
    """
    write_key = os.environ.get("RAINDROP_WRITE_KEY")
    if not write_key:
        _log("RAINDROP_WRITE_KEY absent — running WITHOUT Workshop tracing "
             "(agent still runs; cold open still lands).")
        return None

    counter = CrucibleSpanCounter() if count_spans else None
    try:
        from raindrop_openai_agents import create_raindrop_openai_agents

        client = create_raindrop_openai_agents(
            api_key=write_key,
            user_id=user_id,
            convo_id=convo_id,
            debug=debug,
        )
    except Exception:
        _log("could not create Raindrop bridge (non-fatal):\n" + traceback.format_exc())
        return None

    # EXPLICIT native fan-out: the wrapper already registered the Raindrop
    # processor; we additionally attach our counter + any caller processors via
    # the SDK's add_trace_processor hook (additive — battery #12 proved this).
    attached = []
    for proc in ([counter] if counter else []) + list(extra_processors or []):
        try:
            add_trace_processor(proc)
            attached.append(proc)
        except Exception:
            _log("add_trace_processor failed for a processor (non-fatal):\n"
                 + traceback.format_exc())

    _log(f"bridge live: user_id={user_id!r} convo_id={convo_id!r} "
         f"(+{len(attached)} additive processor(s) via add_trace_processor)")
    return BridgeHandle(client=client, counter=counter, enabled=True, extra=attached)


@contextlib.contextmanager
def crucible_workflow(
    name: str,
    *,
    node: Optional[str] = None,
    crucible_meta: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Iterator[Any]:
    """Wrap an agent run in an SDK ``trace()`` carrying ``crucible.*`` metadata.

    The metadata keys are normalised to the ``crucible.*`` namespace so the SDK
    trace shares context with the domain spans emitted by ``crucible/trace.py``.
    Safe to use even if tracing is disabled (the SDK trace is a cheap no-op).
    """
    meta: dict[str, Any] = {}
    if node:
        meta["crucible.node"] = node
    for k, v in (crucible_meta or {}).items():
        if v is None:
            continue
        meta[k if k.startswith("crucible.") else f"crucible.{k}"] = v
    with trace(workflow_name=name, metadata=meta or None, trace_id=trace_id, group_id=group_id) as t:
        yield t


@contextlib.contextmanager
def crucible_span(name: str, *, data: Optional[dict[str, Any]] = None) -> Iterator[Any]:
    """Emit an explicit ``crucible`` custom span inside an agent run."""
    with custom_span(name=name, data=data or {}) as sp:
        yield sp


__all__ = [
    "install_raindrop_bridge",
    "BridgeHandle",
    "CrucibleSpanCounter",
    "crucible_workflow",
    "crucible_span",
]
