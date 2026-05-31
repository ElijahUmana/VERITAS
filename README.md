# VERITAS — the courtroom for autonomous research

> An agent proposes a confident improvement. **CRUCIBLE** tries it against an
> **external mechanical oracle** in an isolated sandbox. A cheating or wrong claim
> is **REJECTED live**; only a genuinely-verified increment enters a **compounding
> ledger**; run #2 builds on run #1 — and **Raindrop Workshop is the courtroom**
> where every verdict is visible, annotated, and replayable.

We lead with the courtroom. The kernel is the substrate. Our novelty is the **live
cheat-catch + verified compounding ledger + Raindrop audit** — what prior art does
only offline.

---

## The one command

```bash
.venv/bin/python crucible/demo.py
```

That runs the entire FLOOR in order, deterministically, in **under 60 seconds**,
and **verifies itself live** against the local Raindrop Workshop. The cheat and
verified-increment verdicts are produced by the **real CRUCIBLE truth-floor gate**
(`crucible.orchestrator.Orchestrator` + a CPU `ReferenceRMSNormOracle`) — genuine,
gate-produced, not hand-stamped — yet fully deterministic and keyless (no
GPU/Modal/network):

| Beat | What you see | What it proves |
|---|---|---|
| **0–7s · COLD OPEN** | An agent cites two cases. A real case flashes **GREEN** (HTTP 200); a fabricated case flashes **RED** (404 — "this case does not exist"). | Caught a lie in one call, with a database — legible to any judge in 5s. The oracle layer is general/pluggable. |
| **7–22s · THE CHEAT** | A confident "2× faster RMSNorm" is **REJECTED** — the anti-tamper oracle catches the result-reuse reward-hack; a torch-in-disguise cheat is killed by the static pre-gate **before GPU spend**. The Raindrop span goes red with `issue: reward-hack blocked`. | Live, in-the-loop reward-hack rejection. |
| **22–40s · VERIFIED** | An honest Triton RMSNorm passes correctness (5 seeds) + a real dual-timer speedup + anti-tamper → ledger **COMMITTED** with a `proof_hash`; a Claim Certificate is written. | A real verified increment, not an agent's say-so. |
| **40–52s · COMPOUNDING** | **Run #2** reads run #1's verified ledger row as its baseline, skips the already-refuted path, and commits a further gain. | Verified memory that compounds across runs. |
| **52–60s · RAINDROP CLOSE** | A **replay** re-verifies the increment; the demo asserts its own courtroom state on screen and prints the Workshop URLs, the `proof_hash`, and the certificate. | Raindrop is the courtroom — inspectable, annotated, replayable. |

Run modes:

```bash
.venv/bin/python crucible/demo.py            # FLOOR: deterministic CPU gate, <60s, guaranteed green
.venv/bin/python crucible/demo.py --cached   # FLOOR + zero-network cold open (no token/Internet)
.venv/bin/python crucible/demo.py --live     # CEILING: live Modal megastructure — N candidates → N real T4 GPUs at once
.venv/bin/python crucible/demo.py --no-color
```

The default **floor** runs the centerpiece on a deterministic CPU reference oracle
(gate-produced, keyless, <60s, lands every time). `--live` swaps in the real Modal
T4 megastructure: candidates fan out across N distinct live GPUs concurrently, a
reward-hack is caught on real hardware, the honest kernel commits at a real ~2.4×,
and run #2 compounds — every verdict still produced by the same truth-floor gate.

The demo ends with a **TIMING REPORT** (PASS/FAIL on the <60s target) and a **BEAT
SCOREBOARD**. Every beat is verified for real against Workshop — the demo prints
`DEMO GREEN` only if all five beats landed, were confirmed by Workshop readback,
and finished within budget.

---

## The adversarial self-test — "try to break our own demo"

The discipline that keeps the demo honest. It actively attacks the truth floor and
the courtroom and passes **only if every attack is caught**:

```bash
.venv/bin/python tests/adversarial_selftest.py            # all groups
.venv/bin/python tests/adversarial_selftest.py --no-live  # skip the live Workshop group
.venv/bin/python tests/adversarial_selftest.py --quick    # gate + determinism only
.venv/bin/python tests/adversarial_selftest.py --modal    # also run real Modal verification (group D)
```

| Group | What it attacks |
|---|---|
| **A · gate adversarial** | For each cheat/tamper, constructs the verdict the oracle would emit and asserts the §2.3 truth floor **BLOCKS it via the named defense** (correctness, anti-tamper, static, trace-readback/no-oracle, silent-failure, speedup threshold). The honest candidate must **promote**. |
| **B · determinism** | The gate is pure (same inputs → identical verdict, 50×), hashes are stable and order-invariant, the cold-open cache yields the same GREEN/RED twice, and the span/verdict enums agree between `trace.py` and `schemas.py`. |
| **D-static · real static pre-gate** | Runs the **real** static checker on the **real** candidate files (no GPU): the torch-in-disguise cheat is blocked; honest + runtime-only cheats correctly pass through to their runtime judge. |
| **C · live courtroom readback** | Emits a **real** crucible trace, runs the detectors, writes annotations, then asserts via the Workshop query API (the same surface the Raindrop MCP reads) that the **promoted** claim has an oracle span + **no** issue annotation and every **rejected** claim has an issue annotation. Hermetic: isolated event name + cleanup. |
| **D · real Modal** | Pushes each real tamper candidate through the real kernel oracle. Activates when `crucible.oracle.kernel_oracle` lands; until then it **skips loudly** — never a fake pass. |

A failing self-test prints `DO NOT DEMO`.

---

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
cp .env.example .env          # fill OPENAI_API_KEY (optional for the cached floor)
```

The local Raindrop Workshop must be running on `:5899` (it is the courtroom the
demo verifies against). Modal auth (`modal setup`) and `OPENAI_API_KEY` are only
needed for the **live overlays** (ceiling); the `--cached` floor needs neither.

Verify the verification harness itself:

```bash
.venv/bin/python harness/workshop.py     # live Workshop client self-check
.venv/bin/python harness/fallback.py     # cached-fallback layer self-check
.venv/bin/python harness/beats.py        # timing + narration self-check
```

---

## Why it always lands (the backup flow)

FLOOR.md §1: *deterministic backbone, rehearsed inputs, cached fallbacks.* A
WiFi/Modal hiccup cannot kill the run:

- **Cold open** reads hard-cached CourtListener fixtures (`cold_open/cache/`); the
  live call is a best-effort overlay that never changes the GREEN/RED verdict.
- **Cheat / verified beats** run the REAL gate over a CPU reference oracle — genuine
  verdicts, but no GPU/Modal/key, so they're deterministic and can't be killed by a
  network hiccup. The real T4 Modal oracle is the ceiling (proven by the `--modal`
  self-test); if the engine itself is somehow unavailable, the demo falls back to a
  zero-dependency rehearsed courtroom trace.
- **Replay** re-verifies the committed increment in-process via the real gate (or a
  clearly-labelled `deterministic-floor` mode in the fallback) — the response always
  states which path ran; no silent fakery.
- `--cached` forces the zero-network deterministic floor end to end.

---

## Layout (demo + verification surface)

```
crucible/demo.py               # THE one-command <60s demo runner
tests/adversarial_selftest.py  # the "try to break our own demo" self-test
harness/                       # demo + verification support (stdlib-only)
  workshop.py                  #   zero-dep Workshop HTTP client + readback assertions
  fallback.py                  #   record/replay cached-evidence layer
  beats.py                     #   beat timing, <60s budget, colored narration
artifacts/                     # generated Claim Certificates (proof_hash, bounded language)
```

The engine (`crucible/schemas.py`, `trace.py`, `detectors.py`, `replay_server.py`,
the oracle layer, the RMSNorm lab under `benchmarks/`) and the cold open
(`cold_open/`) are built by the rest of the floor team; this demo wires them into
the single guaranteed run and verifies the result live.

---

## Phase-zero integration proofs

The substrate this is built on (OpenAI Agents + Responses, Modal sandbox/GPU/snapshot,
Raindrop OTLP readback) has its own live smoke tests:

```bash
bash phase-zero/modal/run_all.sh
.venv/bin/python phase-zero/openai_live_battery.py --preflight
MODAL_VERIFY_GPU=T4 TRIAD_MODEL=gpt-5.4-mini .venv/bin/python phase-zero/integration/triad_smoke.py
```

The checked-in `.gitignore` blocks `.env`, logs, caches, local virtualenvs, and
generated verification outputs. Keep private strategy docs out of this repo unless
deliberately sanitized.

---

## Closing line

> *"Everyone else builds agents that optimize. We built the courtroom that decides
> whether the optimization is real — and only real, verified increments compound."*
