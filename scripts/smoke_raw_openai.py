"""Live smoke test (a): RAW OpenAI Responses API call.

Run AFTER OPENAI_API_KEY lands in $R/.env:
    .venv/bin/python scripts/smoke_raw_openai.py

Proves: auth works, the Responses API works, gpt-5 reasoning works,
prompt caching params accepted, and LIVE-lists the real model lineup so we
verify model names against the live account (not stale training data).
Writes evidence to logs/smoke_raw_openai.txt and exits non-zero on failure.
"""
from __future__ import annotations

import os
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "smoke_raw_openai.txt"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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


def main() -> int:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not set after loading .env", file=sys.stderr)
        return 2

    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-5")
    ok = True

    print("=== 1. LIVE model list (verify real lineup) ===")
    try:
        ids = sorted(m.id for m in client.models.list().data)
        buckets: dict[str, list[str]] = {
            "gpt": [],
            "o": [],
            "deep-research": [],
            "realtime": [],
            "embedding": [],
            "image": [],
            "audio/tts/transcribe": [],
            "other": [],
        }
        for i in ids:
            if "deep-research" in i:
                buckets["deep-research"].append(i)
            elif "realtime" in i:
                buckets["realtime"].append(i)
            elif "embedding" in i:
                buckets["embedding"].append(i)
            elif "image" in i or "dall-e" in i:
                buckets["image"].append(i)
            elif any(t in i for t in ("tts", "whisper", "transcribe", "audio")):
                buckets["audio/tts/transcribe"].append(i)
            elif i.startswith("gpt"):
                buckets["gpt"].append(i)
            elif i.startswith("o1") or i.startswith("o3") or i.startswith("o4") or i.startswith("o5"):
                buckets["o"].append(i)
            else:
                buckets["other"].append(i)
        print(f"total models visible to this account: {len(ids)}")
        for b, vals in buckets.items():
            if vals:
                print(f"  [{b}] " + ", ".join(vals))
    except Exception:
        ok = False
        print("model list FAILED:\n" + traceback.format_exc())

    print("\n=== 2. Responses API call (gpt-5, reasoning) ===")
    try:
        resp = client.responses.create(
            model=model,
            input="In one sentence, what is the capital of France?",
            reasoning={"effort": "low"},
            max_output_tokens=2000,
            store=True,
            prompt_cache_key="veritas-smoke",
        )
        print("model:", resp.model)
        print("output_text:", resp.output_text)
        print("usage:", resp.usage)
        rid = getattr(resp, "id", None)
        print("response id:", rid)
    except Exception:
        ok = False
        print("Responses call FAILED:\n" + traceback.format_exc())

    print("\n=== 3. Structured output (text.format json_schema) ===")
    try:
        r2 = client.responses.create(
            model=model,
            input="Return the capital and population (approx) of France.",
            reasoning={"effort": "low"},
            max_output_tokens=2000,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "country_fact",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "capital": {"type": "string"},
                            "population_millions": {"type": "number"},
                        },
                        "required": ["capital", "population_millions"],
                    },
                }
            },
        )
        print("structured output_text:", r2.output_text)
    except Exception:
        ok = False
        print("Structured-output call FAILED:\n" + traceback.format_exc())

    print("\n=== RESULT:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.__stdout__, log_file)  # type: ignore[assignment]
        sys.stderr = Tee(sys.__stderr__, log_file)  # type: ignore[assignment]
        try:
            code = main()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr
    raise SystemExit(code)
