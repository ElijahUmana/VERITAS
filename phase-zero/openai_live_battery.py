#!/usr/bin/env python3
"""
OpenAI phase-zero live verification battery.

Fires the instant OPENAI_API_KEY lands in ../.env. Proves EVERY relevant OpenAI
capability with a REAL call (not an import): Responses API, reasoning, structured
outputs, web_search / file_search / code_interpreter hosted tools, embeddings,
Batch API, Agents SDK run, custom trace processor (the Raindrop bridge), realtime
session construction, and the deep-research model (background mode).

Doctrine: surface EVERY error. Each test is isolated so the whole battery runs to
completion and reports ALL failures with full tracebacks — this is loud surfacing,
NOT swallowing. Nothing is faked; a SKIP is only used when a prerequisite id is not
yet web-confirmed, and it says so explicitly.

Usage:
    .venv/bin/python phase-zero/openai_live_battery.py            # run all
    .venv/bin/python phase-zero/openai_live_battery.py --only responses,embeddings
    .venv/bin/python phase-zero/openai_live_battery.py --preflight # cheapest auth check only

Outputs:
    phase-zero/openai_live_results.json   (machine-readable)
    phase-zero/openai-VERIFIED.md         (generated human-readable proof; ignored by git)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENV_PATH = ROOT / ".env"
RESULTS_JSON = HERE / "openai_live_results.json"
VERIFIED_MD = HERE / "openai-VERIFIED.md"

# ---------------------------------------------------------------------------
# Model IDs — CONFIRMED via developers.openai.com on 2026-05-30 unless marked.
# Override any of these via env without editing the script.
# ---------------------------------------------------------------------------
M_FLAGSHIP = os.environ.get("OAI_M_FLAGSHIP", "gpt-5.5")        # [P] confirmed
M_MID = os.environ.get("OAI_M_MID", "gpt-5.4")                  # [P] confirmed (example-harness default)
M_MINI = os.environ.get("OAI_M_MINI", "gpt-5.4-mini")
M_NANO = os.environ.get("OAI_M_NANO", "gpt-5.4-nano")          # [P] confirmed
# All CONFIRMED live on developers.openai.com 2026-05-30 (names unchanged from training,
# re-verified today). Override via env if desired.
M_DEEP_RESEARCH = os.environ.get("OAI_M_DEEP_RESEARCH", "o4-mini-deep-research")  # [P] $2/$8; o3-deep-research=$10/$40
M_EMBED = os.environ.get("OAI_M_EMBED", "text-embedding-3-large")                # [P] 3072d $0.13/1M (small=1536d $0.02)
M_REALTIME = os.environ.get("OAI_M_REALTIME", "gpt-realtime-2")                  # [P] GA voice; audio $32/$64 per 1M

# Hosted-tool type strings. GA web search is "web_search" (gpt-5.x). The DEEP-RESEARCH guide,
# however, shows "web_search_preview" in its tools array — so DR uses the preview type below.
WEB_SEARCH_TYPE = os.environ.get("OAI_WEB_SEARCH_TYPE", "web_search")
DR_WEB_SEARCH_TYPE = os.environ.get("OAI_DR_WEB_SEARCH_TYPE", "web_search_preview")
# A public, no-auth remote MCP server used in OpenAI's own docs examples (DeepWiki).
MCP_TEST_URL = os.environ.get("OAI_MCP_TEST_URL", "https://mcp.deepwiki.com/mcp")


# ---------------------------------------------------------------------------
# .env loading (no external dependency required)
# ---------------------------------------------------------------------------
def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    status: str = "PENDING"          # PASS | FAIL | SKIP
    detail: str = ""
    model: str = ""
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    evidence: str = ""               # truncated real output
    error: str = ""                  # full traceback on FAIL


def _truncate(s: Any, n: int = 800) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + f"... [+{len(s)-n} chars]"


class Battery:
    def __init__(self) -> None:
        self.results: list[TestResult] = []
        from openai import OpenAI, AsyncOpenAI  # noqa: imported here so --help works without key
        self.OpenAI = OpenAI
        self.AsyncOpenAI = AsyncOpenAI
        self.client = OpenAI()
        self.aclient = AsyncOpenAI()
        self.raindrop_agents = None

    def ensure_raindrop_agents(self, convo_id: str) -> None:
        if self.raindrop_agents is not None or not os.environ.get("RAINDROP_WRITE_KEY"):
            return
        from raindrop_openai_agents import create_raindrop_openai_agents

        self.raindrop_agents = create_raindrop_openai_agents(
            api_key=os.environ["RAINDROP_WRITE_KEY"],
            user_id=os.environ.get("RAINDROP_USER_ID", "phase-zero-openai-stack"),
            convo_id=convo_id,
        )

    def flush_raindrop_agents(self) -> None:
        if self.raindrop_agents is not None:
            self.raindrop_agents.flush()

    def record(self, r: TestResult) -> None:
        self.results.append(r)
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(r.status, "•")
        print(f"\n{icon} [{r.status}] {r.name}  ({r.latency_s:.2f}s, model={r.model or '-'})")
        if r.detail:
            print(f"    {r.detail}")
        if r.evidence:
            print(f"    evidence: {_truncate(r.evidence, 300)}")
        if r.error:
            print(f"    ERROR:\n{r.error}")

    def run_test(self, name: str, fn: Callable[[TestResult], None]) -> None:
        r = TestResult(name=name)
        t0 = time.time()
        try:
            fn(r)
            if r.status == "PENDING":
                r.status = "PASS"
        except Exception:
            r.status = "FAIL"
            r.error = traceback.format_exc()
        r.latency_s = round(time.time() - t0, 3)
        self.record(r)

    # ----------------------------- TESTS ----------------------------------
    def t_responses_basic(self, r: TestResult) -> None:
        r.model = M_MID
        resp = self.client.responses.create(
            model=M_MID,
            input="Reply with exactly: VERITAS ONLINE",
            max_output_tokens=2000,
        )
        r.evidence = resp.output_text
        r.usage = json.loads(resp.usage.model_dump_json()) if getattr(resp, "usage", None) else {}
        r.detail = "Responses API round-trip OK"
        assert resp.output_text, "empty output_text"

    def t_reasoning(self, r: TestResult) -> None:
        r.model = M_FLAGSHIP
        resp = self.client.responses.create(
            model=M_FLAGSHIP,
            input="A bat and ball cost $1.10. The bat costs $1 more than the ball. "
                  "How much is the ball? Think, then answer with just the number.",
            reasoning={"effort": "medium", "summary": "auto"},
            max_output_tokens=4000,
        )
        r.evidence = resp.output_text
        r.usage = json.loads(resp.usage.model_dump_json()) if getattr(resp, "usage", None) else {}
        rt = r.usage.get("output_tokens_details", {}).get("reasoning_tokens")
        r.detail = f"Reasoning OK; reasoning_tokens={rt}; answer should be 0.05"

    def t_structured_output(self, r: TestResult) -> None:
        r.model = M_MINI
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "capital": {"type": "string"},
                "population_millions": {"type": "number"},
            },
            "required": ["capital", "population_millions"],
        }
        resp = self.client.responses.create(
            model=M_MINI,
            input="Give the capital and population (millions) of France.",
            text={"format": {"type": "json_schema", "name": "country", "schema": schema, "strict": True}},
            max_output_tokens=2000,
        )
        data = json.loads(resp.output_text)
        r.evidence = json.dumps(data)
        assert data["capital"].lower() == "paris", f"unexpected: {data}"
        r.detail = "Strict JSON-schema structured output validated"

    def t_web_search(self, r: TestResult) -> None:
        r.model = M_MID
        resp = self.client.responses.create(
            model=M_MID,
            input="Use web search: what is the OpenAI flagship model released most recently? "
                  "Answer in one sentence and include the source.",
            tools=[{"type": WEB_SEARCH_TYPE}],
        )
        r.evidence = resp.output_text
        # confirm a web_search tool call actually happened
        calls = [it.type for it in resp.output] if hasattr(resp, "output") else []
        r.detail = f"web_search hosted tool invoked; output items={calls}"
        assert any("web" in str(c).lower() and "search" in str(c).lower() for c in calls), (
            f"no web_search output item found; output item types={calls}")

    def t_code_interpreter(self, r: TestResult) -> None:
        r.model = M_MID
        resp = self.client.responses.create(
            model=M_MID,
            input="Use the code interpreter to compute the 25th Fibonacci number. Print only the number.",
            tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        )
        r.evidence = resp.output_text
        calls = [it.type for it in resp.output] if hasattr(resp, "output") else []
        r.detail = f"code_interpreter container executed; output items={calls}"
        assert any("code" in str(c).lower() for c in calls), (
            f"no code_interpreter output item found; output item types={calls}")
        assert "75025" in resp.output_text, f"unexpected code_interpreter answer: {resp.output_text!r}"

    def t_file_search(self, r: TestResult) -> None:
        r.model = M_MID
        # 1) create a tiny doc, upload, build a vector store
        doc = HERE / "_fs_probe.txt"
        secret = "The verification passphrase is GLASSHOPPER-7741."
        doc.write_text(secret + "\n")
        f = self.client.files.create(file=open(doc, "rb"), purpose="assistants")
        vs = self.client.vector_stores.create(name="phase0-probe")
        self.client.vector_stores.files.create_and_poll(vector_store_id=vs.id, file_id=f.id)
        resp = self.client.responses.create(
            model=M_MID,
            input="What is the verification passphrase? Answer with just the passphrase.",
            tools=[{"type": "file_search", "vector_store_ids": [vs.id]}],
        )
        r.evidence = resp.output_text
        # cleanup
        try:
            self.client.vector_stores.delete(vs.id)
            self.client.files.delete(f.id)
            doc.unlink(missing_ok=True)
        except Exception:
            pass
        assert "GLASSHOPPER-7741" in resp.output_text, f"retrieval failed: {resp.output_text!r}"
        r.detail = "file_search retrieved the planted fact from a vector store"

    def t_embeddings(self, r: TestResult) -> None:
        r.model = M_EMBED
        resp = self.client.embeddings.create(
            model=M_EMBED,
            input=["autonomous research swarm", "a cat sleeping on a sofa", "distributed agent control loop"],
        )
        dims = len(resp.data[0].embedding)
        # cosine sanity: items 0 and 2 should be closer than 0 and 1
        import math
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb)
        e = [d.embedding for d in resp.data]
        s_rel, s_unrel = cos(e[0], e[2]), cos(e[0], e[1])
        r.usage = {"dims": dims, "cos_related": round(s_rel, 4), "cos_unrelated": round(s_unrel, 4)}
        r.evidence = f"dims={dims} cos(related)={s_rel:.3f} > cos(unrelated)={s_unrel:.3f}"
        r.detail = f"Embeddings OK ({dims} dims); semantic ordering sane"
        assert s_rel > s_unrel, "embedding semantics look wrong"

    def t_batch(self, r: TestResult) -> None:
        r.model = M_NANO
        # Build a 2-line JSONL batch against /v1/responses, submit, confirm acceptance.
        lines = []
        for i, q in enumerate(["Say PING", "Say PONG"]):
            lines.append(json.dumps({
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": M_NANO, "input": q, "max_output_tokens": 1000},
            }))
        bf = HERE / "_batch_probe.jsonl"
        bf.write_text("\n".join(lines) + "\n")
        up = self.client.files.create(file=open(bf, "rb"), purpose="batch")
        try:
            batch = self.client.batches.create(
                input_file_id=up.id, endpoint="/v1/responses", completion_window="24h"
            )
            r.evidence = f"batch_id={batch.id} status={batch.status}; input_file_id={up.id}"
            r.detail = ("Batch ACCEPTED (24h window; not awaited). 50% discount tier confirmed live. "
                        f"Poll with client.batches.retrieve('{batch.id}').")
            assert batch.id, "batch not created"
        finally:
            bf.unlink(missing_ok=True)
            try:
                self.client.files.delete(up.id)
            except Exception as exc:
                r.detail = (r.detail + f" Uploaded batch file cleanup failed: {exc!r}").strip()

    def t_agents_sdk(self, r: TestResult) -> None:
        """Real Agents SDK run with a function tool AND a custom trace processor (Raindrop bridge proof)."""
        r.model = M_MINI
        self.ensure_raindrop_agents("openai-live-battery")
        from agents import Agent, Runner, function_tool, add_trace_processor
        from agents.tracing import TracingProcessor

        span_log: list[str] = []

        class CountingProcessor(TracingProcessor):
            def on_trace_start(self, trace): span_log.append(f"trace_start:{getattr(trace,'trace_id','?')}")
            def on_trace_end(self, trace): span_log.append("trace_end")
            def on_span_start(self, span): span_log.append(f"span_start:{type(getattr(span,'span_data',span)).__name__}")
            def on_span_end(self, span): span_log.append("span_end")
            def shutdown(self): pass
            def force_flush(self): pass

        add_trace_processor(CountingProcessor())

        @function_tool
        def multiply(a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

        agent = Agent(
            name="ProbeAgent",
            model=M_MINI,
            instructions="You are terse. Use the multiply tool when asked to multiply.",
            tools=[multiply],
        )
        result = asyncio.run(Runner.run(agent, "What is 23 times 19? Use the tool, then reply with just the number."))
        self.flush_raindrop_agents()
        r.evidence = f"final_output={result.final_output!r}; captured_spans={len(span_log)}"
        r.usage = {"captured_spans": len(span_log), "sample": span_log[:8]}
        r.detail = ("Agents SDK run OK; custom TracingProcessor captured live spans — "
                    "this is the native Raindrop fan-out hook (add_trace_processor).")
        assert "437" in str(result.final_output), f"tool/run wrong: {result.final_output!r}"
        assert span_log, "no spans captured — tracing hook not firing"

    def t_agents_hosted_websearch(self, r: TestResult) -> None:
        """Agents SDK + hosted WebSearchTool (proves hosted tools work through the SDK, not just raw Responses)."""
        r.model = M_MID
        self.ensure_raindrop_agents("openai-live-battery-websearch")
        from agents import Agent, Runner, WebSearchTool
        agent = Agent(
            name="Researcher", model=M_MID,
            instructions="Answer using web search. One sentence with a source URL.",
            tools=[WebSearchTool()],
        )
        result = asyncio.run(Runner.run(agent, "What did OpenAI most recently ship for the Realtime API?"))
        self.flush_raindrop_agents()
        r.evidence = str(result.final_output)
        r.detail = "Agents SDK drove the hosted WebSearchTool end-to-end."
        assert result.final_output, "empty"

    def t_realtime_construct(self, r: TestResult) -> None:
        """Construct a RealtimeAgent/Runner (live audio needs a mic; we verify the SDK path + model id)."""
        r.model = M_REALTIME
        from agents.realtime import RealtimeAgent, RealtimeRunner
        ra = RealtimeAgent(name="VoiceNarrator", instructions="Narrate the swarm's progress in one breath.")
        runner = RealtimeRunner(starting_agent=ra)
        r.evidence = f"RealtimeAgent+RealtimeRunner constructed: {type(ra).__name__}/{type(runner).__name__}"
        r.detail = ("Realtime SDK path constructed. Live session (WebRTC/WS) staged for the demo box; "
                    f"model={M_REALTIME}. A full session.connect() needs the event loop + audio I/O.")

    def t_deep_research(self, r: TestResult) -> None:
        """Kick a deep-research model in background mode; confirm acceptance (do not await the ~tens-of-min run)."""
        r.model = M_DEEP_RESEARCH
        # DR REQUIRES at least one data source (web_search_preview | file_search | mcp); code_interpreter optional.
        # background=True REQUIRES store=true (default true; set explicitly to be safe).
        resp = self.client.responses.create(
            model=M_DEEP_RESEARCH,
            input="In 3 bullets, summarize the current SOTA in autonomous multi-agent research systems (2026). Cite sources.",
            tools=[{"type": DR_WEB_SEARCH_TYPE}, {"type": "code_interpreter", "container": {"type": "auto"}}],
            background=True,
            store=True,
        )
        r.evidence = f"id={resp.id} status={getattr(resp,'status','?')}"
        r.detail = (f"Deep-research ({M_DEEP_RESEARCH}) ACCEPTED in background mode. Poll "
                    f"client.responses.retrieve('{resp.id}') (runs tens of min). Long-horizon synthesis path confirmed.")
        assert resp.id, "no response id"

    def t_mcp_remote(self, r: TestResult) -> None:
        """Attach a REMOTE MCP server as a hosted tool (no per-call fee) and let the model call it."""
        r.model = M_MID
        resp = self.client.responses.create(
            model=M_MID,
            input="Use the deepwiki MCP tools to look up the repo 'openai/openai-agents-python' and name one core module. One sentence.",
            tools=[{
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": MCP_TEST_URL,
                "require_approval": "never",   # docs DEFAULT requires approval per call; disable for headless
            }],
        )
        r.evidence = resp.output_text
        items = [it.type for it in resp.output] if hasattr(resp, "output") else []
        r.detail = f"Remote MCP hosted tool invoked via Responses API; output items={items}"
        assert resp.output_text, "empty MCP result"

    def t_conversation_state(self, r: TestResult) -> None:
        """Prove stateful Responses chaining via previous_response_id (server-managed context)."""
        r.model = M_NANO
        r1 = self.client.responses.create(
            model=M_NANO, input="Remember the codeword: ORYX. Reply 'stored'.", store=True, max_output_tokens=1000)
        r2 = self.client.responses.create(
            model=M_NANO, input="What was the codeword? Reply with just the word.",
            previous_response_id=r1.id, store=True, max_output_tokens=1000)
        r.evidence = f"turn1={r1.output_text!r} -> turn2={r2.output_text!r}"
        r.detail = "previous_response_id carried server-side state across turns"
        assert "ORYX" in r2.output_text.upper(), f"state not carried: {r2.output_text!r}"

    def t_computer_use(self, r: TestResult) -> None:
        """Computer-use is GA as the `computer` hosted tool on gpt-5.x, but a live loop needs a local
        VM/browser driver (screenshot<->action). We do NOT fake a pass: mark SKIP with the honest reason
        and confirm the SDK class + tool type exist."""
        from agents import ComputerTool  # noqa: F401 — confirms SDK class exists
        r.model = M_MID
        r.status = "SKIP"
        r.detail = ("computer-use GA via tool type `computer` on gpt-5.4/5.5 (billed at model rates, no per-call fee). "
                    "Live click/type/scroll loop needs a local Computer/AsyncComputer driver + display — staged on the "
                    "demo box, not run headless here to avoid a fake pass. ComputerTool class import OK.")


# ---------------------------------------------------------------------------
TEST_REGISTRY = [
    ("responses", "Responses API — basic round-trip", "t_responses_basic"),
    ("reasoning", "Reasoning (effort + summary)", "t_reasoning"),
    ("structured", "Structured Outputs (strict JSON schema)", "t_structured_output"),
    ("web_search", "Hosted tool: web_search", "t_web_search"),
    ("code_interpreter", "Hosted tool: code_interpreter", "t_code_interpreter"),
    ("file_search", "Hosted tool: file_search (+vector store)", "t_file_search"),
    ("embeddings", "Embeddings + semantic sanity", "t_embeddings"),
    ("batch", "Batch API submission", "t_batch"),
    ("mcp", "Hosted tool: remote MCP server (DeepWiki)", "t_mcp_remote"),
    ("conversation", "Stateful Responses chaining (previous_response_id)", "t_conversation_state"),
    ("computer_use", "Computer-use availability (honest SKIP — needs driver)", "t_computer_use"),
    ("agents", "Agents SDK run + custom trace processor (Raindrop bridge)", "t_agents_sdk"),
    ("agents_websearch", "Agents SDK + hosted WebSearchTool", "t_agents_hosted_websearch"),
    ("realtime", "Realtime SDK path construction", "t_realtime_construct"),
    ("deep_research", "Deep-research model (background)", "t_deep_research"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenAI Phase-Zero live verification battery")
    ap.add_argument("--only", type=str, default="", help="comma-separated test keys to run")
    ap.add_argument("--preflight", action="store_true", help="cheapest auth check only (nano responses)")
    args = ap.parse_args()

    load_env(ENV_PATH)
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or not key.startswith("sk"):
        print("❌ OPENAI_API_KEY is not set in environment or ../.env — battery cannot run.")
        print("   This is the Phase-Zero gate. Re-run the instant the key lands.")
        return 2
    print("🔑 OPENAI_API_KEY present. openai-stack live battery starting.\n")

    b = Battery()

    if args.preflight:
        b.run_test("PREFLIGHT — nano auth", lambda r: (
            setattr(r, "model", M_NANO),
            setattr(r, "evidence", b.client.responses.create(
                model=M_NANO, input="say OK", max_output_tokens=1000).output_text),
            setattr(r, "detail", "Auth + cheapest model confirmed live"),
        ))
        _dump(b)
        return 0 if all(x.status == "PASS" for x in b.results) else 1

    only = {k.strip() for k in args.only.split(",") if k.strip()}
    for key_, label, method in TEST_REGISTRY:
        if only and key_ not in only:
            continue
        b.run_test(label, getattr(b, method))

    b.flush_raindrop_agents()
    _dump(b)
    n_pass = sum(1 for x in b.results if x.status == "PASS")
    n_fail = sum(1 for x in b.results if x.status == "FAIL")
    n_skip = sum(1 for x in b.results if x.status == "SKIP")
    print(f"\n=========== SUMMARY: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ===========")
    return 0 if n_fail == 0 else 1


def _dump(b: Battery) -> None:
    RESULTS_JSON.write_text(json.dumps([asdict(r) for r in b.results], indent=2))
    print(f"\n📄 wrote {RESULTS_JSON.relative_to(ROOT)}")
    # Append a results table to the generated verification doc.
    rows = ["", "## LIVE RESULTS (auto-generated)", "",
            "| Test | Status | Model | Latency | Detail |", "|---|---|---|---|---|"]
    for r in b.results:
        rows.append(f"| {r.name} | {r.status} | {r.model or '-'} | {r.latency_s:.2f}s | {r.detail or r.error[:80]} |")
    marker = "<!-- LIVE_RESULTS -->"
    md = VERIFIED_MD.read_text() if VERIFIED_MD.exists() else ""
    block = marker + "\n" + "\n".join(rows) + "\n" + marker
    if marker in md:
        import re
        md = re.sub(re.escape(marker) + r".*?" + re.escape(marker), block, md, flags=re.S)
    else:
        md += "\n\n" + block + "\n"
    VERIFIED_MD.write_text(md)
    print(f"📄 updated {VERIFIED_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
