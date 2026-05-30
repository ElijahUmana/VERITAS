"""Live smoke test (b): minimal OpenAI Agents SDK agent.

Run AFTER OPENAI_API_KEY lands in $R/.env:
    .venv/bin/python scripts/smoke_agents_sdk.py

Proves: the Agents SDK harness works end-to-end on gpt-5 — the agent loop,
a custom function_tool round-trip, structured output via output_type, and the
hosted web_search tool (the core of our research pipeline). Exits non-zero on
failure.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        print("FATAL: $R/.env not found — key has not landed yet.", file=sys.stderr)
        sys.exit(2)
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def usage_of(result) -> str:
    try:
        u = result.context_wrapper.usage
        return f"requests={u.requests} input={u.input_tokens} output={u.output_tokens} total={u.total_tokens}"
    except Exception:
        return "<usage unavailable>"


async def main() -> int:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not set after loading .env", file=sys.stderr)
        return 2

    from pydantic import BaseModel

    from agents import Agent, Runner, WebSearchTool, function_tool

    raindrop_client = None
    if os.environ.get("RAINDROP_WRITE_KEY"):
        from raindrop_openai_agents import create_raindrop_openai_agents

        raindrop_client = create_raindrop_openai_agents(
            api_key=os.environ["RAINDROP_WRITE_KEY"],
            user_id=os.environ.get("RAINDROP_USER_ID", "scripts-smoke-agents-sdk"),
            convo_id=os.environ.get("RAINDROP_CONVO_ID", "veritas"),
        )

    ok = True

    # --- 1. function_tool loop ---
    print("=== 1. Agent + function_tool loop ===")
    calls: list[str] = []

    @function_tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers.

        Args:
            a: first integer
            b: second integer
        """
        calls.append(f"multiply({a},{b})")
        return a * b

    try:
        agent = Agent(
            name="SmokeCalc",
            instructions="You are a precise calculator. ALWAYS use the multiply tool for products.",
            tools=[multiply],
        )
        result = await Runner.run(agent, "What is 23 times 19? Use the tool.")
        print("final_output:", result.final_output)
        print("tool calls observed:", calls)
        print("usage:", usage_of(result))
        print("last_agent:", result.last_agent.name)
        if "437" not in str(result.final_output) or not calls:
            ok = False
            print("WARN: expected 437 via a tool call")
    except Exception:
        ok = False
        print("function_tool agent FAILED:\n" + traceback.format_exc())

    # --- 2. structured output via output_type ---
    print("\n=== 2. Structured output via output_type ===")

    class CityFact(BaseModel):
        city: str
        country: str
        is_capital: bool

    try:
        agent2 = Agent(
            name="SmokeStruct",
            instructions="Answer with the structured type.",
            output_type=CityFact,
        )
        r2 = await Runner.run(agent2, "Tell me about Paris.")
        print("structured final_output:", r2.final_output)
        print("usage:", usage_of(r2))
    except Exception:
        ok = False
        print("structured-output agent FAILED:\n" + traceback.format_exc())

    # --- 3. hosted web_search tool (core research capability) ---
    print("\n=== 3. Hosted web_search tool ===")
    try:
        researcher = Agent(
            name="SmokeResearcher",
            instructions="Use web search to answer with a current fact and cite the source.",
            tools=[WebSearchTool()],
        )
        r3 = await Runner.run(
            researcher,
            "Using web search, what is the latest OpenAI flagship model announced? One sentence + source.",
        )
        print("research final_output:", r3.final_output)
        print("usage:", usage_of(r3))
    except Exception:
        ok = False
        print("web_search agent FAILED:\n" + traceback.format_exc())

    print("\n=== RESULT:", "PASS" if ok else "FAIL", "===")
    if raindrop_client is not None:
        raindrop_client.flush()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
