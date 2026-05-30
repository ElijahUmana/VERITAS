# VERITAS

VERITAS is a phase-zero scaffold for verified autonomous research systems.
Its first public surface focuses on live, reproducible integration proofs across:

- OpenAI Agents and Responses API smoke tests
- Modal sandbox, GPU, snapshot, volume, pool, and autoscaler verification
- Raindrop monitoring integration for OpenAI Agents traces
- A triad smoke test joining OpenAI orchestration, Modal execution, and Raindrop readback

This repository intentionally excludes private planning notes, research memos,
logs, credentials, local virtual environments, and vendor checkouts.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
cp .env.example .env
```

Fill `.env` with local credentials:

```bash
OPENAI_API_KEY=
RAINDROP_WRITE_KEY=
RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/
MODAL_VERIFY_GPU=T4
```

Authenticate Modal separately:

```bash
modal setup
```

## Smoke Tests

Run the Modal phase-zero suite:

```bash
bash phase-zero/modal/run_all.sh
```

Run the OpenAI preflight:

```bash
.venv/bin/python phase-zero/openai_live_battery.py --preflight
```

Run the OpenAI Agents smoke:

```bash
.venv/bin/python scripts/smoke_agents_sdk.py
```

Run the triad proof:

```bash
MODAL_VERIFY_GPU=T4 TRIAD_MODEL=gpt-5.4-mini .venv/bin/python phase-zero/integration/triad_smoke.py
```

## Safety

The checked-in `.gitignore` blocks `.env`, logs, caches, local virtualenvs,
and generated verification outputs. Keep private strategy docs and raw research
notes out of this repository unless they are deliberately sanitized.
