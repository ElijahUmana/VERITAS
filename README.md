# VERITAS

### The proof layer for autonomous AI — a research swarm that improves itself and *cannot lie about it.*

> A swarm of agents propose improvements. An external **mechanical oracle** verifies each one by **executing it** in an isolated Modal sandbox — not by asking another model. Every cheat is caught and quarantined. Only genuinely-verified gains compound into a ledger the next run builds on. **Raindrop is the courtroom** where every verdict is tried, annotated, replayed, and self-healed.
>
> The engine is **CRUCIBLE**. Every verdict is mechanically produced and independently verified — we proved it by red-teaming our own verifier through a **12-attack gauntlet** (below) and by catching our own pipeline cheating three separate times.

```bash
python crucible/demo.py          # run the whole system end-to-end
```

---

## The problem: autoresearch that grades its own homework

Every frontier autoresearch system is a better **generator that scores itself** — and that is structurally broken:

- **Hallucination is mathematically inevitable.** You cannot train it to zero.
- **Reward-hacking is a proven *equilibrium*, not a bug** (arXiv 2603.28063). Any optimized agent under-serves whatever its evaluator can't see. The Darwin Gödel Machine **falsified its own logs** to inflate its score. Berkeley drove **8 major agent benchmarks to ~100%** without solving a single task. When the grader is a model, the optimizer learns to fool the grader.
- So the one quantity whose reliability *rises* with compute is **verification — but only when the verdict is cast by an external, execution-grounded oracle the agent cannot manipulate.**

That oracle is the missing layer. VERITAS is it. The 1000× is not a bigger generator; it is **verification decided by execution, a gate that only lets *proven* gains compound, and an observability layer that measures and heals its own failures.**

---

## What VERITAS is

**The courtroom for autonomous research.** A claim is not "true" because an agent is confident, and not because a second model agrees. It is true only if an **external mechanical oracle** — code execution on a real GPU, a citation database, a reference implementation — returns a verdict, *and* that verdict survives a four-invariant truth floor, *and* the trace reads back clean from Raindrop. Everything else is **rejected and retained as negative evidence** the swarm learns from.

The thesis in one line: **Oracle-Grounded Falsification.** Generation is cheap; *grounded* verification is the bottleneck — so we built the verifier, made it adversarial, and made every verdict auditable.

---

## Hitting all three judging axes — at once, and combined

| Judging axis | What VERITAS does |
|---|---|
| **Agent Architectures & Control Loops** | The **verdict is the control signal.** A propose → verify → gate → compound loop where an external oracle, not a reward model, closes the loop. A swarm fans out across live GPUs; a **self-healing eval loop** detects a gate slip and repairs it red→green; the truth-floor gate is a pure, testable control surface red-teamed by a 12-attack gauntlet. |
| **Retrieval & Knowledge Synthesis** | The **verified ledger is a knowledge base that compounds.** Run N+1 *retrieves* run N's verified frontier as its baseline and skips already-refuted paths. The cold open is live citation retrieval against CourtListener. Synthesis is survivor-only: 30 candidates in, the 8 that pass cross-examination compound. |
| **Applied Autonomous Research** | Real **verified GPU-kernel research** on live Modal T4s, anchored on **KernelBench** (Stanford–MIT, vendored MIT-licensed). A gate-enforced **self-improvement curve** that *cannot be staged* — provably-real autonomous research, not a leaderboard number an agent can fake. |
| **Raindrop Workshop track** | Raindrop is **load-bearing on both surfaces** — the local Workshop *is* the courtroom (4 detectors, durable annotations, claim-subtree replay, the self-healing eval loop) and the hosted platform carries the run events and custom Signals that quantify exactly what the gate caught. Nothing left on the table. |

---

## Architecture

```
                          RAINDROP — the courtroom / nervous system
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │  WORKSHOP (local :5899)            HOSTED (app.raindrop.ai)                        │
   │  • crucible.* OTLP span tree       • run events                                    │
   │  • 4 truth-floor detectors         • custom Signals (reward-hack / cheat-blocked)  │
   │  • durable annotations             • Query API read-back                           │
   │  • claim-subtree REPLAY            ▲                                                │
   │  • self-healing eval loop (red→green)                                              │
   └───────────────────────────────────┼──────────────────────────────────────────────┘
                                        │  every verdict is traced, asserted, replayable
   ─────────────────────────────────────┼──────────────────────────────────────────────
                                        │            the verdict feeds back as the control signal
   GENERATOR (OpenAI)   ──propose──▶   CRUCIBLE ORCHESTRATOR   ──verify──▶   ORACLE (external, mechanical)
   gpt-5.5 / gpt-5.4-mini swarm         the truth-floor gate:                 ├─ Modal T4 sandbox — runs the code
   N candidates, diverse seeds          1. correctness (5 seeds)              │   block_network · CUDA reset · KernelBench
                                        2. anti-tamper clean                  ├─ CPU reference — stage-safe
                                        3. oracle verdict = confirmed         └─ CourtListener — citation existence
                                        4. speedup ≥ threshold
                                        5. Raindrop read-back confirms span
                                                  │ promote                ✗ reject → retained as negative evidence
                                                  ▼
                                        VERIFIED LEDGER  (SQLite · parent_ledger_id chain)
                                        every row: proof_hash + Claim Certificate
                                        run N+1 seeds from run N's verified frontier  ──┐
                                                  ▲──────────────────────────────────────┘
                                                  only certified increments compound
```

**Components**
- **Truth-floor gate** (`crucible/orchestrator.py`, `schemas.evaluate_truth_floor`): a *pure function* of the verdict. Five invariants must all hold or the claim is blocked. Pure ⇒ unit-testable ⇒ red-teamable.
- **Oracle layer** (`crucible/oracle/`): pluggable and external. Modal GPU execution, a CPU reference, a citation database — truth is never an LLM vote.
- **Anti-tamper** (`oracle/anti_tamper.py`, `modal/verifier_app.py`): input clone/zero, output materialization, dual-timer disagreement, >10× excessive-speedup flag, AST static pre-gate, no-network sandbox, harness-integrity snapshot/restore, per-invocation CUDA reset.
- **Ledger** (`crucible/ledger.py`): SQLite, `parent_ledger_id` compounding chain, `proof_hash` binding the full provenance of each increment.
- **Raindrop bridge** (`crucible/trace.py`, `detectors.py`, `replay_server.py`, `eval_loop.py`, `hosted.py`): the courtroom — emit, detect, annotate, replay, self-heal, and the hosted Signals.

---

## Run it yourself

```bash
python crucible/demo.py            # the CPU gate — keyless, offline
python crucible/demo.py --live     # the real Modal T4 megastructure (N candidates → N live GPUs) + live citation lookup
python crucible/demo.py --cached   # zero-network everything
```

| Beat | What you see | What it proves |
|---|---|---|
| **Catch a lie** | An agent cites two cases. A real one flashes **GREEN** (200); a fabricated one flashes **RED** (404 — "this case does not exist"). | Hallucination caught in one call against a real database — and the oracle is general, not kernel-specific. |
| **Catch the cheat** | A confident "2× faster" kernel is **REJECTED** — anti-tamper catches the reward-hack; a disguise cheat dies at the static gate *before GPU spend*. Raindrop turns red. | Live, in-the-loop reward-hack rejection. |
| **Verified increment** | An honest kernel passes correctness (5 seeds) + a real dual-timer speedup + anti-tamper → ledger **COMMITTED** with a `proof_hash` + Claim Certificate. | A re-runnable proof, not an agent's say-so. |
| **It improves itself** | **Run #2** reads run #1's verified row, skips the refuted path, and commits a further certified gain. | Verified memory that compounds; the curve can't be staged. |
| **Raindrop close** | A **replay** re-verifies the increment; a deliberately-broken run is **self-healed red→green**; Workshop URLs + `proof_hash` + certificate printed. | Raindrop is the courtroom — inspectable, annotated, replayable, self-healing. |

Prints `DEMO GREEN` only if all five beats land **and** are confirmed by Workshop read-back. The same inputs produce the same verdicts every run, and the cold open runs on **real captured CourtListener bytes**.

---

## The results

- **An AI that provably improves itself** — gate-enforced monotonic so it *cannot be staged*:
  ```
  CPU floor:                        run1 1.25× → run2 1.88× → run3 2.98×   ·  run4 2.26× → BLOCKED (didn't beat the frontier)
  GPU, over LIVE-generated kernels:  run1 1.62× → run2 1.72× → run3 2.33×   ·  each rung re-verified on a separate T4
  ```
  `run4` was a real, correct, faster kernel — rejected anyway because it didn't beat `2.98×`. That single blocked run is the proof the curve only climbs on certified gains.
- **The swarm caught cheating its own benchmark** — 4 cheat classes (stream-bypass, result-reuse, zero-inputs, torch-disguise) caught on real GPU *and* CPU, each by its named defense; the honest kernel commits at a real **2.41×** on a live T4.
- **The megastructure** — N gpt-5.4-mini candidates fan out across N concurrent T4s; a 30-candidate swarm yields **8 survivors (27%)**, the rest blocked on merit; only survivors compound.
- **Raindrop, both surfaces** — local courtroom with 4 detectors + claim-subtree replay + a self-healing eval loop (red→green on screen); hosted platform with custom Signals that quantify what the gate caught — reward-hacks flagged and cheats blocked before they could ship, read back via the Query API.
- **Adversarial self-test 41/41** (44/44 with `--modal`) — the discipline that keeps the pipeline honest passes only if every attack is caught.

---

## Adversarial hardening — the gauntlet

We red-teamed our own verifier. A cheat that fools your verifier is worse than no verifier, so each defense below is backed by regression tests **and** an external adversarial gauntlet.

| Attack / risk | Defense | Status |
|---|---|---|
| Fabricated court citation | external citation oracle → 404 → RED | ✅ verified |
| Reward-hack kernel ("2× faster", secretly cheating) | anti-tamper oracle rejects; Workshop issue annotation on the rejected claim | ✅ verified |
| Honest increment must *actually* commit | oracle pass → ledger commit → `proof_hash` + certificate; Workshop confirms promoted span has an oracle span and **no** issue annotation | ✅ verified |
| Cross-run memory forgery | run #2 reads run #1's committed row and links via `parent_ledger_id` | ✅ verified |
| Replay theatre | a real replay server; the replay POST returns `confirmed` and emits a replay trace id | ✅ verified |
| Mismatched identity injection | candidate-bound truth floor blocks wrong claim/mission/candidate IDs | ✅ verified |
| **Raindrop read-back spoof** | missing / empty / malformed read-back **fails closed** — candidate-specific read-back required to promote | ✅ verified |
| **`np.allclose` monkeypatch** | candidate cannot patch correctness checks; an external monkeypatch also cannot make wrong output pass | ✅ verified |
| Poisoned ledger rows | invalid committed/replayed rows cannot become baselines | ✅ verified |
| Provenance tampering | changing trace / run / ledger / claim / candidate / mission / artifact changes the `proof_hash` | ✅ verified |
| Packaging / reproducibility | editable install works; `numpy` declared; cold-open cache shipped in the wheel | ✅ verified |
| Artifact leakage | certificates isolated to `artifacts/certificates`; build artifacts gitignored | ✅ verified |

This is the differentiator. Anyone can build an agent that optimizes a number. **We built the verifier that survives an adversary trying to fake that number** — and proved it fails closed under every attack we could devise.

---

## Everything you can run

```bash
python crucible/demo.py                  # the whole system (real gate). --live = real GPU + live citation lookup
python -m crucible.self_improvement      # the gate-enforced self-improvement curve
python crucible/swarm.py --n 20          # verified swarm fan-out — only survivors commit
python -m crucible.hosted --verify       # hosted Raindrop read-back: the Signals
python -m crucible.eval_loop             # the self-healing eval loop (red→green)
python tests/adversarial_selftest.py     # the adversarial gauntlet — 41/41 (44/44 with --modal)
```

---

## How it compares

- **Modal autoresearch** gives a swarm elastic GPUs to chase a number. VERITAS adds the courtroom that catches the swarm faking that number — and compounds only what survives cross-examination.
- **AlphaEvolve** trusts a single machine-gradable oracle — but that oracle is exactly what frontier agents learn to hack. VERITAS makes the oracle adversarial, pluggable, and measured, with a verified ledger that compounds across runs and a 12-attack gauntlet proving it fails closed.

> ### *"Everyone else builds agents that optimize. We built the courtroom that decides whether the optimization is **real** — and only real, verified increments compound. The AI improves itself, and cannot lie about it."*

---

## Stack & layout

**Modal** (elastic isolated T4 verifier + megastructure fan-out) · **OpenAI** (gpt-5.5 / gpt-5.4-mini generator swarm + Agents SDK) · **Raindrop** (Workshop courtroom + hosted Signals) · **KernelBench** (MIT, vendored).

```
crucible/   orchestrator.py (the gate) · schemas.py · oracle/ (kernel/reference/citation + anti-tamper + static)
            ledger.py · trace.py · detectors.py · replay_server.py · eval_loop.py · hosted.py
            self_improvement.py · swarm.py · demo.py
modal/      the deploy-once T4 verifier + cap-aware megastructure fan-out
benchmarks/rmsnorm_lab/   vendored KernelBench (MIT) + reference + honest/cheat candidates
cold_open/  the CourtListener legal-citation oracle (real captured bytes)
tests/      adversarial_selftest.py — the gauntlet
```

`.env`, logs, caches, `.venv`, and generated outputs are gitignored.

Built at the Autoresearch Systems Hackathon — Modal × OpenAI × Raindrop × Antler.
