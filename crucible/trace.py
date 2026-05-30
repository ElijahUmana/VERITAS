#!/usr/bin/env python3
"""crucible/trace.py — VERITAS / CRUCIBLE direct OTLP emitter for Raindrop Workshop.

Implements the ``crucible.*`` span contract (FLOOR.md §2.1). Emits OTLP/JSON
straight to the local Workshop daemon at ``$RAINDROP_LOCAL_DEBUGGER`` (default
``http://localhost:5899/v1/traces``). No SDK, no API key — this is the
SDK-version-proof local path proven in ``research/raindrop.md`` §6 and
``sandbox/otlp_proof.py``.

Span typing uses ``raindrop.span.kind`` (the tag Workshop's ``parse.ts`` honors
first): ``agent_root`` → AGENT_ROOT, ``llm_call`` → LLM_GENERATION,
``tool_call`` → TOOL_CALL, ``trace`` → TRACE.

The courtroom (``crucible/detectors.py``) reads these spans back via the
Workshop query API / raindrop MCP ``query_traces``, so the attribute names below
are a HARD contract shared with crucible-core's orchestrator. Co-designed for
Task #3 (raindrop-courtroom) ⇄ Task #1 (crucible-core).

Encoding decisions (verified against the live daemon):
  * bool attrs ``correctness_passed`` / ``tamper_detected`` are emitted as
    **int 0/1** so the §2.3 promotion gate (``==1`` / ``==0``) and detector C
    (``tamper_detected=1``) work directly under SQLite ``json_extract``.
  * custom ``crucible.*`` keys land verbatim in the spans ``attributes`` JSON
    column, queryable via ``json_extract(attributes,'$."crucible.X"')``.
"""
from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
import urllib.request

__all__ = [
    "CrucibleTracer", "Span", "BASE", "TRACES_PATH",
    "NODES", "ORACLE_TYPES", "VERDICTS", "PROMOTIONS", "SPAN_KINDS",
]


# --- daemon base URL -------------------------------------------------------
def _base_url() -> str:
    raw = os.environ.get("RAINDROP_LOCAL_DEBUGGER", "http://localhost:5899/v1/").rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw  # -> http://localhost:5899


BASE = _base_url()
TRACES_PATH = "/v1/traces"


# --- OTLP value encoders ---------------------------------------------------
def _s(v):  return {"stringValue": str(v)}
def _i(v):  return {"intValue": str(int(v))}
def _d(v):  return {"doubleValue": float(v)}
def _kv(k, v): return {"key": k, "value": v}


# --- crucible.* contract enums (validated; fail-fast, no silent drops) -----
NODES = {"mission", "claim", "candidate", "verify", "oracle", "anti_tamper", "ledger", "replay"}
ORACLE_TYPES = {"correctness", "speed", "anti_tamper", "replay", "citation"}
VERDICTS = {"confirmed", "refuted", "blocked", "unverified"}
PROMOTIONS = {"committed", "blocked", "replayed", "regressed"}
SPAN_KINDS = {"agent_root", "trace", "llm_call", "tool_call"}
_STATUS_CODE = {"UNSET": 0, "OK": 1, "ERROR": 2}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ns(ms: int) -> str:
    return str(int(ms) * 1_000_000)


# --- HTTP ------------------------------------------------------------------
def _post(url, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach Workshop daemon at {url}: {e}") from e


class Span:
    """A single in-flight crucible span.

    Created by :meth:`CrucibleTracer.span`. Call :meth:`finish` to stamp the end
    time + status and register it for the next :meth:`CrucibleTracer.flush`.
    """

    def __init__(self, tracer, span_id, parent_id, name, kind, node,
                 claim_id, candidate_id, oracle_type, start_ms):
        self._tracer = tracer
        self.span_id = span_id
        self.parent_id = parent_id
        self.name = name
        self.node = node
        self.start_ms = start_ms
        self._finished = False
        self._attrs = list(tracer._assoc)  # event metadata on every span
        self._attrs.append(_kv("raindrop.span.kind", _s(kind)))
        self._attrs.append(_kv("crucible.mission_id", _s(tracer.mission_id)))
        self._attrs.append(_kv("crucible.node", _s(node)))
        if claim_id is not None:
            self._attrs.append(_kv("crucible.claim_id", _s(claim_id)))
        if candidate_id is not None:
            self._attrs.append(_kv("crucible.candidate_id", _s(candidate_id)))
        if oracle_type is not None:
            self._attrs.append(_kv("crucible.oracle_type", _s(oracle_type)))

    # -- attribute application (the shared kwarg → crucible.* map) -----------
    def _apply(self, attrs):
        for k, v in attrs.items():
            if v is None:
                continue
            if k == "verdict":
                if v not in VERDICTS:
                    raise ValueError(f"verdict must be one of {sorted(VERDICTS)}, got {v!r}")
                self._attrs.append(_kv("crucible.verdict", _s(v)))
            elif k == "promotion":
                if v not in PROMOTIONS:
                    raise ValueError(f"promotion must be one of {sorted(PROMOTIONS)}, got {v!r}")
                self._attrs.append(_kv("crucible.promotion", _s(v)))
            elif k == "oracle_type":
                if v not in ORACLE_TYPES:
                    raise ValueError(f"oracle_type must be one of {sorted(ORACLE_TYPES)}, got {v!r}")
                self._attrs.append(_kv("crucible.oracle_type", _s(v)))
            elif k == "correctness_passed":
                self._attrs.append(_kv("crucible.correctness_passed", _i(1 if v else 0)))
            elif k == "tamper_detected":
                self._attrs.append(_kv("crucible.tamper_detected", _i(1 if v else 0)))
            elif k == "speedup":
                self._attrs.append(_kv("crucible.speedup", _d(v)))
            elif k == "confidence":
                self._attrs.append(_kv("crucible.confidence", _d(v)))
            elif k == "blocked_reason":
                self._attrs.append(_kv("crucible.blocked_reason", _s(v)))
            elif k == "ledger_id":
                self._attrs.append(_kv("crucible.ledger_id", _s(v)))
            elif k == "candidate_id":
                self._attrs.append(_kv("crucible.candidate_id", _s(v)))
            elif k == "input":
                self._attrs.append(_kv("traceloop.entity.input", _s(v if isinstance(v, str) else json.dumps(v))))
            elif k == "output":
                self._attrs.append(_kv("traceloop.entity.output", _s(v if isinstance(v, str) else json.dumps(v))))
            elif k == "tool_name":
                self._attrs.append(_kv("tool.name", _s(v)))
            elif k == "tool_input":
                self._attrs.append(_kv("tool.input", _s(v if isinstance(v, str) else json.dumps(v))))
            elif k == "tool_output":
                self._attrs.append(_kv("tool.output", _s(v if isinstance(v, str) else json.dumps(v))))
            elif k == "model":
                self._attrs.append(_kv("gen_ai.request.model", _s(v)))
            elif k == "provider":
                self._attrs.append(_kv("gen_ai.system", _s(v)))
            elif k == "input_tokens":
                self._attrs.append(_kv("gen_ai.usage.input_tokens", _i(v)))
            elif k == "output_tokens":
                self._attrs.append(_kv("gen_ai.usage.output_tokens", _i(v)))
            elif k.startswith("crucible_"):
                # generic crucible.* passthrough (string-valued)
                self._attrs.append(_kv("crucible." + k[len("crucible_"):], _s(v)))
            else:
                raise ValueError(f"unknown span attribute {k!r} (not in crucible.* contract)")
        return self

    def set(self, **attrs):
        """Attach attributes without finishing the span."""
        return self._apply(attrs)

    def finish(self, status="OK", end_ms=None, **attrs):
        """Stamp end time + status, apply any final attrs, register for flush."""
        if self._finished:
            raise RuntimeError(f"span {self.name!r} already finished")
        if status not in _STATUS_CODE:
            raise ValueError(f"status must be one of {list(_STATUS_CODE)}, got {status!r}")
        if attrs:
            self._apply(attrs)
        end = end_ms if end_ms is not None else max(_now_ms(), self.start_ms + 1)
        otlp = {
            "traceId": self._tracer.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "startTimeUnixNano": _ns(self.start_ms),
            "endTimeUnixNano": _ns(end),
            "status": {"code": _STATUS_CODE[status]},
            "attributes": self._attrs,
        }
        if self.parent_id:
            otlp["parentSpanId"] = self.parent_id
        self._tracer._spans.append(otlp)
        self._finished = True
        return self


class CrucibleTracer:
    """Accumulates ``crucible.*`` spans for one mission and flushes them as a
    single OTLP/JSON batch to the local Workshop daemon.

    One :class:`CrucibleTracer` == one Workshop run (one ``trace_id``).
    """

    def __init__(self, mission_id, event_name="veritas_crucible",
                 user_id="veritas", convo_id="autoresearch-hackathon",
                 service_name="veritas-crucible", base=None, trace_id=None,
                 replay_run_id=None, event_id=None):
        self.mission_id = mission_id
        self.trace_id = trace_id or secrets.token_hex(16)
        self.event_name = event_name
        # When replaying, the event_id IS the replayRunId so the begin/finish-style
        # stitch key matches too (see setup-agent-replay Python guidance).
        self.event_id = event_id or replay_run_id or f"{event_name}-{secrets.token_hex(4)}"
        self.user_id = user_id
        self.convo_id = convo_id
        self.service_name = service_name
        self.base = (base or BASE).rstrip("/")
        self.replay_run_id = replay_run_id
        self._spans = []
        # event-association metadata so the run carries event_name/user/convo + the
        # event_id stitch key (parse.ts reads traceloop.association.properties.*)
        self._assoc = [
            _kv("traceloop.association.properties.event_id", _s(self.event_id)),
            _kv("traceloop.association.properties.event_name", _s(event_name)),
            _kv("traceloop.association.properties.user_id", _s(user_id)),
            _kv("traceloop.association.properties.convo_id", _s(convo_id)),
        ]
        # replay stitch key (§2.5/§2.6): when this run IS a replay, tag every span
        # via ALL three surfaces Workshop checks so it stitches the fresh trace to
        # the placeholder replay run: the raindrop.replayRunId attr, the traceloop
        # replayRunId property, and event_id (set above).
        if replay_run_id:
            self._assoc.append(_kv("raindrop.replayRunId", _s(replay_run_id)))
            self._assoc.append(_kv("traceloop.association.properties.replayRunId", _s(replay_run_id)))

    def span(self, node, kind, name, claim_id=None, candidate_id=None,
             oracle_type=None, parent=None, start_ms=None, **attrs) -> Span:
        if node not in NODES:
            raise ValueError(f"node must be one of {sorted(NODES)}, got {node!r}")
        if kind not in SPAN_KINDS:
            raise ValueError(f"kind must be one of {sorted(SPAN_KINDS)}, got {kind!r}")
        if oracle_type is not None and oracle_type not in ORACLE_TYPES:
            raise ValueError(f"oracle_type must be one of {sorted(ORACLE_TYPES)}, got {oracle_type!r}")
        parent_id = parent.span_id if isinstance(parent, Span) else parent
        sp = Span(self, secrets.token_hex(8), parent_id, name, kind, node,
                  claim_id, candidate_id, oracle_type, start_ms if start_ms is not None else _now_ms())
        if attrs:
            sp.set(**attrs)
        return sp

    def flush(self):
        """POST all finished spans as one OTLP batch. Returns the ingest JSON."""
        if not self._spans:
            return None
        payload = {"resourceSpans": [{
            "resource": {"attributes": [_kv("service.name", _s(self.service_name))]},
            "scopeSpans": [{
                "scope": {"name": "veritas-crucible", "version": "1.0"},
                "spans": list(self._spans),
            }],
        }]}
        st, body = _post(self.base + TRACES_PATH, payload)
        if st != 200:
            raise RuntimeError(f"OTLP ingest failed ({st}): {body}")
        return json.loads(body) if body else {"status": st}

    @property
    def run_url(self) -> str:
        return f"{self.base}/runs/{self.trace_id}"


if __name__ == "__main__":
    # Non-emitting self-description (does not pollute Workshop).
    print("crucible/trace.py — crucible.* OTLP span contract (FLOOR §2.1)")
    print(f"  daemon base    = {BASE}")
    print(f"  ingest path    = {TRACES_PATH}")
    print(f"  nodes          = {sorted(NODES)}")
    print(f"  oracle_types   = {sorted(ORACLE_TYPES)}")
    print(f"  verdicts       = {sorted(VERDICTS)}")
    print(f"  promotions     = {sorted(PROMOTIONS)}")
    print(f"  span kinds     = {sorted(SPAN_KINDS)}")
    print("  run with the demo emitter (crucible/courtroom_demo.py) to emit a trace.")
