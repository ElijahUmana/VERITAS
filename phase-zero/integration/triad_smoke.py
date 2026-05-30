#!/usr/bin/env python
"""Phase-Zero triad capstone: OpenAI brain, Modal GPU body, Raindrop nervous system.

Proves the whole VERITAS integration path end to end in one motion:
  THINK   — an OpenAI Agents SDK `SandboxAgent` decides what to do.
  ACT     — its shell tool executes inside a Modal GPU sandbox (ModalSandboxClient + SandboxRunConfig);
            the agent runs `nvidia-smi` on real GPU hardware and writes/reads a file there.
  TRACE   — `add_trace_processor(<bridge>)` fans spans (agent/generation/function) into
            Raindrop Workshop (local, keyless, :5899) AND the cloud (if RAINDROP_WRITE_KEY is set).

Run:  .venv/bin/python phase-zero/integration/triad_smoke.py
Env:  MODAL_VERIFY_GPU=T4   TRIAD_MODEL=gpt-5.4-mini   RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/
"""
from __future__ import annotations
import os, sys, json, asyncio, pathlib, traceback, time
from datetime import datetime, timezone

_REPO = pathlib.Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO / ".env")
except Exception:
    pass

GPU = os.environ.get("MODAL_VERIFY_GPU", "T4")
MODEL = os.environ.get("TRIAD_MODEL", "gpt-5.4-mini")
APP = os.environ.get("TRIAD_APP", "veritas-triad")
WORKSHOP_ORIGIN = os.environ.get("TRIAD_WORKSHOP_ORIGIN", "http://localhost:5899").rstrip("/")
READBACK_USER_ID = os.environ.get("TRIAD_RAINDROP_USER_ID", "phase-zero-triad")
READBACK_CONVO_ID = os.environ.get("TRIAD_RAINDROP_CONVO_ID", "veritas")
READBACK_EVENT_NAME = "Agent workflow"
READBACK_TIMEOUT_S = float(os.environ.get("TRIAD_WORKSHOP_READBACK_TIMEOUT_S", "15"))
# Ensure the Raindrop SDK mirrors to the already-running local Workshop (keyless).
os.environ.setdefault("RAINDROP_LOCAL_DEBUGGER", "http://localhost:5899/v1/")


def ts() -> str: return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def epoch_ms() -> int: return int(datetime.now(timezone.utc).timestamp() * 1000)
def log(m: str) -> None: print(f"[{ts()}] {m}", flush=True)
def ok(m: str) -> None: print(f"[{ts()}] \033[32mPASS\033[0m  {m}", flush=True)
def fail(m: str) -> None: print(f"[{ts()}] \033[31mFAIL\033[0m  {m}", flush=True)


def _redact_secrets(text: str) -> str:
    for key in ("OPENAI_API_KEY", "RAINDROP_WRITE_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        value = os.environ.get(key)
        if value:
            text = text.replace(value, "<redacted>")
    return text


def _safe_exc(exc: BaseException) -> str:
    return _redact_secrets(f"{type(exc).__name__}: {exc}")


def preflight() -> list[str]:
    """Return a list of missing prerequisites (empty == ready to run live)."""
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY (.env) — blocks the OpenAI brain")
    if not (pathlib.Path.home() / ".modal.toml").exists():
        missing.append("~/.modal.toml (modal setup) — blocks the Modal GPU body")
    # Raindrop local Workshop is keyless; just note if the daemon is unreachable.
    try:
        import urllib.request
        urllib.request.urlopen(f"{WORKSHOP_ORIGIN}/", timeout=2)
    except Exception:
        missing.append(f"Raindrop Workshop daemon at {WORKSHOP_ORIGIN} unreachable (raindrop workshop start)")
    return missing


def validate_model_before_modal() -> None:
    """Fail before provisioning a Modal sandbox if TRIAD_MODEL is invalid."""
    from openai import OpenAI

    try:
        OpenAI().models.retrieve(MODEL)
    except Exception as exc:
        raise RuntimeError(f"TRIAD_MODEL {MODEL!r} failed OpenAI model preflight") from exc
    ok(f"OpenAI model preflight resolved {MODEL!r}")


def install_raindrop_bridge():
    """Install the Raindrop trace processor. Prefer the official OpenAI Agents
    integration when the cloud write key is present; fall back to a local span-capturing
    processor so the triad still proves trace fan-out in keyless Workshop mode.
    Returns (processor, source_str)."""
    from agents import add_trace_processor
    from agents.tracing import TracingProcessor

    # 1) Preferred: Raindrop's official OpenAI Agents SDK integration. It registers
    # its processor globally and captures agent runs, model calls, and tool spans.
    if os.environ.get("RAINDROP_WRITE_KEY"):
        try:
            from raindrop_openai_agents import create_raindrop_openai_agents

            client = create_raindrop_openai_agents(
                api_key=os.environ["RAINDROP_WRITE_KEY"],
                user_id=READBACK_USER_ID,
                convo_id=READBACK_CONVO_ID,
            )
            return client, "raindrop-openai-agents official bridge"
        except Exception:
            log("official Raindrop bridge failed to initialize; falling back to local processor")

    # 2) Fallback: capture spans locally and mirror a compact summary to Workshop.
    captured = {"traces": 0, "spans": [], "raindrop_runs": []}

    class _FallbackRaindropProcessor(TracingProcessor):
        def on_trace_start(self, trace): captured["traces"] += 1
        def on_trace_end(self, trace):
            # Mirror a compact summary into Workshop so a run is visible end-to-end.
            try:
                import raindrop.analytics as rd

                rd.init(
                    api_key=os.environ.get("RAINDROP_WRITE_KEY"),
                    tracing_enabled=False,
                )
                inter = rd.begin(
                    user_id=READBACK_USER_ID,
                    event="veritas_triad_smoke",
                    convo_id=READBACK_CONVO_ID,
                    properties={"model": MODEL, "spans": len(captured["spans"])},
                    input="triad smoke",
                )
                inter.finish(output=f"captured {len(captured['spans'])} span(s)")
                rd.flush()
                captured["raindrop_runs"].append(getattr(inter, "event_id", "?"))
            except Exception as e:
                captured["raindrop_runs"].append(f"<mirror-skip: {_safe_exc(e)}>")
        def on_span_start(self, span): pass
        def on_span_end(self, span):
            captured["spans"].append(type(getattr(span, "span_data", span)).__name__)
        def shutdown(self): pass
        def force_flush(self): pass

    proc = _FallbackRaindropProcessor()
    proc._captured = captured  # type: ignore[attr-defined]
    add_trace_processor(proc)
    return proc, "fallback local processor"


def flush_raindrop(proc) -> None:
    try:
        from agents.tracing import flush_traces
        flush_traces()
    except Exception as exc:
        log(f"agents.tracing.flush_traces skipped after error: {_safe_exc(exc)}")

    for name in ("flush", "force_flush"):
        flush = getattr(proc, name, None)
        if not callable(flush):
            continue
        try:
            flush()
        except Exception as exc:
            log(f"Raindrop {name} skipped after error: {_safe_exc(exc)}")

    processor = getattr(proc, "processor", None)
    for name in ("force_flush", "flush"):
        flush = getattr(processor, name, None)
        if not callable(flush):
            continue
        try:
            flush()
        except Exception as exc:
            log(f"Raindrop processor {name} skipped after error: {_safe_exc(exc)}")

    try:
        import raindrop.analytics as rd
        rd.flush()
    except Exception as exc:
        log(f"raindrop.analytics.flush skipped after error: {_safe_exc(exc)}")


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _workshop_query(sql: str, *, limit: int = 10, max_bytes: int = 20000) -> dict:
    """Run a read-only SQL query against the local Workshop daemon.

    Newer Workshop exposes /api/traces/query. Keep /query first as a cheap
    compatibility probe for local builds that expose the shorter endpoint.
    """
    import urllib.error
    import urllib.request

    payload = json.dumps({"sql": sql, "limit": limit, "max_bytes": max_bytes}).encode()
    errors: list[str] = []
    for path in ("/query", "/api/traces/query"):
        req = urllib.request.Request(
            f"{WORKSHOP_ORIGIN}{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = resp.read().decode()
            data = json.loads(body)
            if isinstance(data, dict):
                data["_endpoint"] = path
                return data
            raise RuntimeError(f"{path} returned non-object JSON")
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405):
                errors.append(f"{path}: HTTP {exc.code}")
                continue
            try:
                body = _redact_secrets(exc.read().decode()[:240])
            except Exception:
                body = ""
            errors.append(f"{path}: HTTP {exc.code} {body}")
        except Exception as exc:
            errors.append(f"{path}: {_safe_exc(exc)}")
    raise RuntimeError("; ".join(errors))


async def run_triad() -> dict:
    from agents import Runner
    # RunConfig / SandboxRunConfig live under agents.run_config (defensive import).
    try:
        from agents import RunConfig
    except Exception:
        from agents.run_config import RunConfig  # type: ignore
    from agents.run_config import SandboxRunConfig
    from agents.sandbox import SandboxAgent
    from agents.extensions.sandbox.modal import ModalSandboxClient, ModalSandboxClientOptions

    log(f"building Modal sandbox client (app={APP}, gpu={GPU}) ...")
    client = ModalSandboxClient()
    options = ModalSandboxClientOptions(app_name=APP, gpu=GPU)

    agent = SandboxAgent(
        name="triad-gpu-probe",
        model=MODEL,
        instructions=(
            "You are running on a remote GPU machine. Do EXACTLY this, using your shell tool:\n"
            "1. Run `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` and read the GPU name.\n"
            "2. Write that GPU name to /root/triad_gpu.txt, then `cat /root/triad_gpu.txt` to confirm.\n"
            "3. Reply with ONLY the GPU name string you observed (e.g. 'Tesla T4')."
        ),
    )

    log(f"Runner.run — SandboxAgent on Modal GPU sandbox, model={MODEL} ...")
    result = await Runner.run(
        agent,
        "Confirm the GPU you are running on and report its exact name.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(client=client, options=options),
            workflow_name="veritas_triad_smoke",
            trace_metadata={
                "user_id": READBACK_USER_ID,
                "convo_id": READBACK_CONVO_ID,
                "modal_gpu": GPU,
            },
        ),
    )
    out = (result.final_output or "").strip()
    log(f"agent final_output: {out!r}")
    return {"final_output": out}


def verify_workshop_run(started_after_ms: int) -> dict:
    """Exact local Workshop readback for the triad trace.

    The official bridge owns its own processor internals, so the only useful
    proof is Workshop's persisted rows: one fresh run joined to at least one
    span, matched by the triad user/conversation identifiers or the Agents SDK
    default event name.
    """
    info: dict[str, object] = {
        "ok": False,
        "origin": WORKSHOP_ORIGIN,
        "started_after_ms": started_after_ms,
        "rows": [],
    }
    try:
        import urllib.request

        with urllib.request.urlopen(f"{WORKSHOP_ORIGIN}/health", timeout=2) as resp:
            info["health"] = json.loads(resp.read().decode())
    except Exception as exc:
        info["error"] = f"Workshop health check failed: {_safe_exc(exc)}"
        return info

    exact_user = _sql_literal(READBACK_USER_ID)
    exact_convo = _sql_literal(READBACK_CONVO_ID)
    sql = f"""
SELECT
  r.id,
  r.event_name,
  r.user_id,
  r.convo_id,
  r.started_at,
  r.last_updated_at,
  COUNT(s.id) AS span_count,
  MIN(s.name) AS first_span_name,
  'triad_ids' AS match_kind
FROM runs r
JOIN spans s ON s.run_id = r.id
WHERE r.last_updated_at >= {int(started_after_ms)}
  AND r.user_id = {exact_user}
  AND r.convo_id = {exact_convo}
GROUP BY r.id, r.event_name, r.user_id, r.convo_id, r.started_at, r.last_updated_at
ORDER BY r.last_updated_at DESC
""".strip()
    info["query"] = "fresh triad run/span by exact user_id+convo_id"

    deadline = time.monotonic() + READBACK_TIMEOUT_S
    last_error = None
    while True:
        try:
            data = _workshop_query(sql, limit=5)
            rows = data.get("rows", [])
            info["endpoint"] = data.get("_endpoint")
            info["row_count"] = data.get("row_count")
            info["rows"] = rows if isinstance(rows, list) else []
            fresh_rows = []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    try:
                        span_count = int(row.get("span_count") or 0)
                    except Exception:
                        span_count = 0
                    if span_count >= 1:
                        fresh_rows.append(row)
            if fresh_rows:
                info["match_kinds"] = sorted({
                    str(row.get("match_kind")) for row in fresh_rows
                })
                info["ok"] = True
                return info
        except Exception as exc:
            last_error = _safe_exc(exc)
            info["error"] = last_error
        if time.monotonic() >= deadline:
            if last_error:
                info["error"] = last_error
            return info
        time.sleep(0.5)


def main() -> int:
    print("=" * 72)
    print(f"TRIAD CAPSTONE (#10) :: OpenAI[{MODEL}] · Modal GPU[{GPU}] · Raindrop[:5899]")
    print("=" * 72)

    missing = preflight()
    if missing:
        fail("NOT READY — staged, awaiting prerequisites:")
        for m in missing:
            print(f"    ⧗ {m}")
        print("\nThis is the EXPECTED state until #1/#4/#6 land. Re-run when they do.")
        return 2  # distinct code: 'blocked', not 'broken'

    try:
        validate_model_before_modal()
    except Exception as exc:
        fail(_safe_exc(exc))
        return 1

    proc, src = install_raindrop_bridge()
    log(f"Raindrop trace bridge installed: {src}")

    run_started_ms = epoch_ms() - 1000
    try:
        triad = asyncio.run(run_triad())
    except Exception as exc:
        fail(f"triad run errored: {_safe_exc(exc)}")
        print(_redact_secrets(traceback.format_exc()), flush=True)
        flush_raindrop(proc)
        return 1

    # Assertion 1: the brain+body produced a GPU-grounded answer.
    out = triad["final_output"]
    fam = GPU.split(":")[0].split("-")[0].upper()
    brain_body_ok = bool(out) and fam in out.upper()
    (ok if brain_body_ok else fail)(
        f"THINK+ACT: agent ran on Modal GPU and reported {out!r} (expected family {fam!r})")

    # Assertion 2: the nervous system captured the trajectory.
    cap = getattr(proc, "_captured", None)
    official_bridge = cap is None and "official bridge" in src
    if cap is not None:
        spans, traces, rdruns = len(cap["spans"]), cap["traces"], cap["raindrop_runs"]
        log(f"TRACE: {traces} trace(s), {spans} span(s) captured: {sorted(set(cap['spans']))}; "
            f"workshop runs mirrored: {rdruns}")
        trace_ok = traces >= 1 and spans >= 1
    else:
        log("TRACE: using official Raindrop bridge — local Workshop read-back is the span gate.")
        trace_ok = False

    flush_raindrop(proc)
    ws = verify_workshop_run(run_started_ms)
    log(f"Workshop read-back: {ws}")

    if official_bridge:
        trace_ok = bool(ws.get("ok"))
        (ok if trace_ok else fail)(
            "TRACE: official bridge produced at least one fresh Workshop run/span")
    else:
        (ok if trace_ok else fail)("TRACE: spans fanned to the Raindrop processor")
        (ok if ws.get("ok") else fail)(
            "TRACE READBACK: local Workshop has a fresh triad run/span"
            if ws.get("ok")
            else "TRACE READBACK: no fresh triad run/span found in local Workshop (non-gating fallback path)"
        )

    if brain_body_ok and trace_ok:
        ok("TRIAD VERIFIED LIVE — OpenAI brain → Modal GPU body → Raindrop nervous system, end to end.")
        return 0
    fail("TRIAD INCOMPLETE — see failed assertion(s) above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
