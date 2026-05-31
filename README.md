# VERITAS

### The proof layer for autonomous AI — an agent swarm that improves itself and *can't lie about it.*

> A swarm of AI agents propose improvements. An external **mechanical oracle** verifies each one in an isolated Modal sandbox — code execution, not LLM opinion. Every cheat is caught and quarantined. Only genuinely-verified increments compound into a ledger the next run builds on. **Raindrop is the courtroom** where every verdict is tried, replayed, and healed.
>
> The engine is called **CRUCIBLE.** Every verdict in this repo is *mechanically produced and independently MCP-verified* — none is hand-stamped. We proved that by repeatedly catching our own demo cheating (three times) and hardening it each time.

```bash
python crucible/demo.py          # the whole thing, <60s, deterministic, guaranteed-green
```

---

## Why this exists

Every frontier autoresearch system is a better **generator that grades itself** — and that's structurally broken:

- **Hallucination is mathematically inevitable** — you can't train it away.
- **Reward-hacking is a proven *equilibrium*, not a bug** (arXiv 2603.28063): any optimized agent under-serves what its evaluator can't see, and the *only* escape is *external ground-truth verification not subject to agent manipulation.* The Darwin Gödel Machine literally falsified its own logs to game its score; Berkeley broke 8 major agent benchmarks to ~100% without solving a single task.
- So the bottleneck — and the one thing whose reliability *rises* with compute — is **verification, but only when the verdict is cast by an external, model-independent oracle.**

VERITAS builds exactly that escape. The 1000× isn't a bigger generator; it's the missing layer underneath: **verification decided by execution, a gate that only lets *proven* gains compound, and an observability layer that measures and heals its own failures.**

---

## It's real — the verified evidence

Every number below is gate-produced and confirmed by reading the trace back through the Raindrop MCP / Query API. Reproduce any of it with the commands in [Run everything](#run-everything).

**An AI that provably improves itself** — the self-improvement curve, gate-enforced monotonic so it *cannot be staged*:

```
CPU floor:                       run1 1.25× → run2 1.88× → run3 2.98×   ·  run4 2.26× → BLOCKED (didn't beat the frontier)
GPU, over LIVE-generated kernels: run1 1.62× → run2 1.72× → run3 2.33×   ·  each rung re-verified on a separate T4 oracle
```
Each run's promotion threshold is *the prior verified best.* `run4` was a **real, correct, faster** kernel — rejected anyway because it didn't beat `2.98×`. That single blocked run is the proof the curve only climbs on certified gains.

**Catch the swarm cheating its own benchmark** — 4 cheat classes, caught on real GPU *and* CPU, distinguished from honest mistakes:

| Candidate | Oracle verdict | Anti-tamper | Result |
|---|---|---|---|
| honest fused RMSNorm | confirmed, **2.41× measured** (real T4) | clean | **committed** (proof_hash + certificate) |
| stream-bypass cheat | refuted (impossible 2644× flagged) | **tamper=1** | blocked |
| result-reuse cheat | refuted (stale-memory recompute mismatch) | **tamper=1** | blocked |
| zero-inputs cheat | refuted (mutated its input in place) | **tamper=1** | blocked |
| wrong-axis kernel | refuted (`max_abs_err` > tol) | tamper=0 *(a mistake, not a cheat)* | blocked |
| torch-in-disguise | static pre-gate (`code_bypass`) | — | blocked, **zero GPU spend** |

**The megastructure** — N gpt-5.4-mini-generated candidates fan out across N concurrent real Modal T4 GPUs; only verified survivors compound (a 30-candidate swarm → **8 survivors / 27%**, the rest blocked on merit).

**Raindrop, load-bearing on both surfaces:**
- *Local Workshop courtroom* — 4 truth-floor detectors, durable annotations, claim-subtree replay, and a **self-healing eval loop** (catches a gate slip → writes an eval → re-verifies → **red→green** on screen).
- *Hosted cloud platform* — events, 6 custom Signals, and a real A/B experiment proving anti-tamper is load-bearing: **cheats shipped lax-oracle = 5, strict-oracle = 0**, read back via the Query API.

**The demo is honest because we kept catching ourselves:** a canned-verdict showpiece (rewired to the real gate), a 25% CPU-timing false-refute (root-caused + fixed, now 10/10), and a GPU co-tenancy state-bleed where a verdict depended on which candidate ran before it (fixed with a per-invocation CUDA reset). The integrity culture *is* the product.

---

## See it — the `<60s` demo

```bash
python crucible/demo.py            # deterministic CPU gate, <60s, guaranteed-green, keyless
python crucible/demo.py --live     # the real Modal T4 megastructure (N candidates → N live GPUs at once)
python crucible/demo.py --cached   # zero-network: even the cold open runs from cache
```

| Beat | What you see | What it proves |
|---|---|---|
| **0–7s · Catch a lie** | An agent cites two cases. A real one flashes **GREEN** (200); a fabricated one flashes **RED** (404 — "this case does not exist"). | Caught a hallucination in one call — with a database, not an opinion. The oracle layer is general/pluggable. |
| **7–22s · Catch the cheat** | A confident "2× faster" kernel is **REJECTED** — the anti-tamper oracle catches the reward-hack; a disguise cheat dies at the static gate *before GPU spend.* Raindrop turns red. | Live, in-the-loop reward-hack rejection. |
| **22–40s · Verified increment** | An honest kernel passes correctness (5 seeds) + a real dual-timer speedup + anti-tamper → ledger **COMMITTED** with a `proof_hash` + Claim Certificate. | A re-runnable proof, not an agent's say-so. |
| **40–52s · It improves itself** | **Run #2** reads run #1's verified row, skips the refuted path, and commits a further certified gain — the climbing curve. | Verified memory that compounds; the curve can't be staged. |
| **52–60s · Raindrop close** | A **replay** re-verifies the increment; a deliberately-broken run is **self-healed red→green**; Workshop URLs + `proof_hash` + certificate printed. | Raindrop is the courtroom — inspectable, annotated, replayable, self-healing. |

The run ends with a **timing report** (PASS/FAIL on `<60s`) and a **beat scoreboard**, and prints `DEMO GREEN` only if all five beats landed *and were confirmed by Workshop readback.* Verified **10/10** across repeated runs.

---

## How it works

```
                         RAINDROP — the courtroom / nervous system
        traces · detectors · annotations · replay · self-healing eval · hosted Signals/Experiments
                                   ▲ verdicts feed back as the control signal
   ─────────────────────────────────┼───────────────────────────────────────────────────────
   GENERATOR (OpenAI)   ──propose──▶ │  CRUCIBLE ORCHESTRATOR (the truth-floor gate)  ──verify──▶  ORACLE
   gpt-5.5 / gpt-5.4-mini swarm       │  assign id → oracle → spans → readback → gate → ledger      Modal T4 sandbox (real GPU)
                                      │            ▼                                                 OR deterministic CPU reference
                                      │     VERIFIED LEDGER (SQLite; parent_ledger_id chain)         OR CourtListener existence check
                                      │     only certified increments compound across runs
```

- **The truth-floor gate** (`crucible/orchestrator.py`): a claim is promoted *only if* correctness passes, anti-tamper is clean, the oracle verdict is `confirmed`, the speedup clears the threshold, and a Workshop readback confirms the oracle span exists. Anything else is **blocked and retained as negative evidence.**
- **The oracle is external and mechanical** — and *pluggable*: a Modal GPU sandbox (execution), a CPU reference (deterministic/stage-safe), a CourtListener existence check (citations). Truth is never an LLM vote.
- **Anti-tamper** (`crucible/oracle/anti_tamper.py`, `modal/verifier_app.py`): input clone/zero, output materialization, dual-timer disagreement, `>10×` excessive-speedup flag, a static pre-gate, a no-network sandbox, harness-integrity snapshots, and per-invocation CUDA reset. We vendor **KernelBench** (MIT) verbatim for the timing + static checks — maximum credibility.
- **The ledger** (`crucible/ledger.py`): SQLite, `parent_ledger_id` chain, every row a `proof_hash` + certificate; run N+1 seeds from run N's verified frontier.
- **Dual-mode:** the deterministic CPU path is the guaranteed stage floor (zero GPU/network); `--live` swaps in the real Modal T4 megastructure. Same gate, same span contract, both MCP-verified.

---

## Run everything

```bash
python crucible/demo.py                  # the <60s demo (real gate, deterministic). --live for real GPU, --cached for zero-network
python -m crucible.self_improvement      # the gate-enforced self-improvement curve (CPU floor)
python crucible/swarm.py --n 20          # verified swarm fan-out — N candidates, only survivors commit
python -m crucible.hosted --verify       # hosted Raindrop read-back: Signals + the strict-vs-lax A/B
python -m crucible.eval_loop             # the self-healing eval loop (red→green)
python -m crucible.spine_acceptance      # engine spine acceptance (23/23)
python tests/adversarial_selftest.py     # "try to break our own demo" — 41/41 (44/44 with --modal)
```

The **adversarial self-test** is the discipline that keeps it honest — it attacks the gate and passes only if every attack is caught:

| Group | What it attacks |
|---|---|
| **A · gate** | Every cheat/tamper must be BLOCKED by its named defense; the honest candidate must promote. |
| **B · determinism** | Pure gate (same inputs → identical verdict ×50), stable hashes, deterministic cold open, enum agreement. |
| **C · live courtroom** | Emits a real trace, runs the detectors, asserts via Workshop readback: promoted ⇒ oracle span + no issue; rejected ⇒ issue. |
| **D · static / Modal** | The real static checker on real candidate files (no GPU); `--modal` pushes real tampers through the live T4 oracle. |
| **E · floor reliability** | The honest candidate commits 10/10 through the real gate, and `run_eval(run)["passed"]` holds — a guard so the floor can never silently flake-regress. |

A failing self-test prints `DO NOT DEMO`.

---

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
cp .env.example .env     # OPENAI_API_KEY, RAINDROP_WRITE_KEY, RAINDROP_QUERY_API_KEY (all optional for the cached floor)
```

The local **Raindrop Workshop** runs on `:5899` (the courtroom the demo verifies against). Modal auth (`modal setup`) + `OPENAI_API_KEY` are only needed for the **live** overlays; the deterministic floor needs neither. The phase-zero substrate (Modal GPU, OpenAI Agents SDK, Raindrop OTLP) has its own live smoke tests under `phase-zero/`.

---

## Repo layout

```
crucible/
  orchestrator.py     # the truth-floor gate (the heart)
  schemas.py          # Claim / Candidate / Verdict / Certificate / LedgerRow + evaluate_truth_floor()
  oracle/             # the pluggable oracle layer (kernel / reference / citation) + anti-tamper + static checker
  ledger.py           # SQLite verified ledger, parent_ledger_id compounding chain
  trace.py            # crucible.* OTLP span contract (the Raindrop emitter)
  detectors.py        # the 4 truth-floor detectors + annotations
  replay_server.py    # Workshop replay of a claim's verification subtree
  eval_loop.py        # the self-healing eval loop (red→green)
  hosted.py           # hosted Raindrop platform (Signals + Experiments + read-back)
  self_improvement.py # the gate-enforced self-improvement curve
  swarm.py            # parallel verified swarm fan-out
  demo.py             # the one-command <60s demo
modal/                # the real Modal T4 verifier (deploy-once) + megastructure fan-out
benchmarks/rmsnorm_lab/  # vendored KernelBench (MIT) + reference + honest/cheat candidates
cold_open/            # the CourtListener legal-citation cold open (cached + live)
tests/adversarial_selftest.py   # "try to break our own demo"
harness/              # stdlib-only Workshop client + cached-fallback + beat timing
phase-zero/           # live integration proofs for the substrate
SUBMISSION.md         # demo script + writeup + positioning
```

`.env`, logs, caches, `.venv`, and generated outputs are gitignored.

---

## Why it wins

- **vs Modal autoresearch:** *"They gave a swarm elastic GPUs to chase a number. We built the courtroom that catches the swarm faking that number and compounds only what survives cross-examination."*
- **vs AlphaEvolve:** *"It trusts a single machine-gradable oracle — but the oracle is exactly what frontier agents learn to hack. We make the oracle adversarial, pluggable, and measured, with a verified ledger that compounds across runs."*

> ### *"Everyone else builds agents that optimize. We built the courtroom that decides whether the optimization is **real** — and only real, verified increments compound. The AI improves itself, and can't lie about it."*

Built at the Autoresearch Systems Hackathon — Modal × OpenAI × Raindrop × Antler.
