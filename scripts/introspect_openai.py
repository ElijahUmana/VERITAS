"""Offline introspection of installed openai + openai-agents libraries.
No network/API key required — inspects signatures and types only.
Cross-checks web/source research against what we can actually call.
"""
from __future__ import annotations

import inspect


def banner(label: str) -> None:
    print(f"\n=== {label} ===")


# --- openai version ---
import openai

banner("openai version")
print(openai.__version__)

# --- ReasoningEffort literal values ---
banner("ReasoningEffort literal source")
try:
    from openai.types.shared import reasoning_effort as _re

    print(inspect.getsource(_re).strip()[:400])
except Exception as e:  # noqa: BLE001
    print("ERR:", type(e).__name__, e)

# --- service_tier / verbosity annotations on responses.create ---
import openai.resources.responses as R

sig = inspect.signature(R.Responses.create)
for p in ("service_tier", "verbosity", "truncation", "reasoning_effort"):
    banner(f"responses.create param: {p}")
    param = sig.parameters.get(p)
    print(str(param.annotation) if param else "<absent>")

# --- hosted tool param types in openai.types.responses ---
banner("openai.types.responses *ToolParam")
import openai.types.responses as resp

print(", ".join(sorted([n for n in dir(resp) if n.endswith("ToolParam")])))

# --- Reasoning type fields ---
banner("openai.types.Reasoning fields")
try:
    from openai.types import Reasoning

    print(list(getattr(Reasoning, "model_fields", {}).keys()))
except Exception as e:  # noqa: BLE001
    print("ERR:", type(e).__name__, e)

# --- Agents SDK Runner surface ---
banner("agents.Runner.run signature")
from agents import Runner

print(str(inspect.signature(Runner.run)))

banner("agents hosted tool classes present")
import agents

print(
    ", ".join(
        sorted(
            n
            for n in dir(agents)
            if n.endswith("Tool")
            or n in {"function_tool", "handoff", "WebSearchTool", "FileSearchTool"}
        )
    )
)

# --- WebSearchTool / FileSearchTool / CodeInterpreterTool fields ---
for tname in (
    "WebSearchTool",
    "FileSearchTool",
    "CodeInterpreterTool",
    "ImageGenerationTool",
    "ComputerTool",
    "LocalShellTool",
    "HostedMCPTool",
):
    banner(f"agents.{tname} fields")
    try:
        cls = getattr(agents, tname)
        flds = getattr(cls, "__dataclass_fields__", None)
        print(list(flds.keys()) if flds else "<not a dataclass>")
    except Exception as e:  # noqa: BLE001
        print("ERR:", type(e).__name__, e)

# --- default model + settings ---
banner("agents default model + GPT-5 default settings")
try:
    from agents.models.default_models import get_default_model, get_default_model_settings

    print("default model:", get_default_model())
    print("default settings:", get_default_model_settings())
except Exception as e:  # noqa: BLE001
    print("ERR:", type(e).__name__, e)
