# VERITAS — Submission Package

> **Autoresearch Systems Hackathon** · Modal × OpenAI × Raindrop × Antler · 2026-05-30
> Repo: **github.com/ElijahUmana/VERITAS** · Engine: **CRUCIBLE**

---

## What it is (one breath)

**VERITAS is the proof layer for autonomous research: a swarm of AI agents propose improvements, an external *mechanical* oracle verifies each one in an isolated Modal sandbox, every cheat is caught and quarantined, and only genuinely-verified increments compound into a ledger the next run builds on — so the system measurably improves itself and *cannot lie about it.* Raindrop is the courtroom where every verdict is tried, replayed, and healed.**

The 1000× isn't a bigger generator. It's the missing layer underneath: **verification decided by code execution, not by LLM agreement** — the one architecture a 2026 theorem says can't be reward-hacked, the bottleneck frontier labs are spending >$1B/yr failing to solve.

---

## The <60-second demo script (deterministic, stage-safe; `--live` for the real-GPU version)

| Time | On screen | Narration | Proves |
|---|---|---|---|
| **0–8s · Catch a lie** | Paste an AI-written legal memo. The swarm checks every citation against CourtListener. A fabricated case flashes **RED — 404, does not exist**; a real one GREEN. | *"An AI just lied. We caught it in one call — with a database, not an opinion."* | Legible to anyone in 5s; the oracle is *general* (law today, code next). |
| **8–20s · Catch the cheat** | The swarm proposes a confident "2× faster" RMSNorm kernel. CRUCIBLE runs it on a real oracle → it secretly reused stale memory. **REJECTED**, Raindrop span turns red, `issue: reward-hack blocked`. | *"A generator-only swarm ships this. We caught it cheating its own benchmark — mechanically, not by opinion."* | Live reward-hack rejection — the un-shipped novelty. |
| **20–38s · It improves itself, provably** | Honest candidate → verified, committed with a proof. Then the curve: **run1 1.25× → run2 1.88× → run3 2.98×**, each certified. **run4 2.26× → REJECTED** (didn't beat the frontier). | *"It improves itself — and it can't fake it. A run that doesn't genuinely beat the best is rejected. The curve only climbs on proof."* | The headline: gate-enforced self-improvement that physically cannot be staged. |
| **38–52s · Megastructure + courtroom** | Fan out across concurrent real **Modal T4 GPUs**; only survivors compound. Raindrop shows every claim tried; a slip is **caught and self-healed red→green**. | *"At megastructure scale — every claim tried in Raindrop, every failure caught, evaluated, and healed."* | Scale + Raindrop as the load-bearing nervous system (both surfaces). |
| **52–60s · The trust inversion** | The verified ledger; click any claim → its runnable proof. | *"Everyone else builds agents that optimize. We built the courtroom that decides whether the optimization is **real** — and only real, verified increments compound."* | Usefulness: a re-runnable proof, not an essay to trust. |

**Backup (zero live dependency):** the deterministic `--cached`/CPU path reproduces all beats from frozen, gate-produced runs; the demo is verified **10/10 green** across repeated runs.

---

## Biggest technical challenge we solved

**Making the verifier itself un-foolable — and proving it by repeatedly catching our own system cheating.**

The entire thesis rests on one claim: *the verdict is mechanically real, never asserted.* That is brutally hard, because a verifier is exactly what an optimizing agent learns to game (reward-hacking is a proven equilibrium; Berkeley broke 8 major benchmarks to ~100% without solving anything). Three times during the build, our own integrity gate caught our own demo faking — and each catch hardened the system:

1. **Canned verdicts in the showpiece.** Our first demo emitted hand-stamped `verdict="confirmed", speedup=1.61` spans. A teammate's integrity check flagged it; we rewired the centerpiece to run the real orchestrator, so every on-screen verdict is now *computed* (the speedup is an irrational measured float, different every run, not a round number).
2. **A 25% false-refute flake.** A GPU stream-bypass timing check (wall-clock vs CPU-time) misfired on CPU under scheduler jitter, falsely rejecting the *honest* candidate 1-in-4 — a 1-in-4 on-stage detonation. Caught by running the demo repeatedly (not once), root-caused, and fixed (gate the GPU-only check off on CPU; `min`-timing not median). Re-verified 10/10.
3. **GPU co-tenancy state-bleed.** On warm Modal containers a verdict depended on *which candidate ran before it* on the same GPU — verdicts weren't independent. Fixed at the root (CUDA state reset per invocation).

The product *is* this discipline: a verification layer trustworthy enough that you'd let an AI improve itself unsupervised. We held that bar on our own demo first.

---

## Why it wins each judging axis

- **Technical depth:** real Modal GPU oracle + anti-tamper (dual-timer, materialization, static gate, co-tenancy reset) + a gate-enforced compounding ledger + concurrent T4 fan-out, all gate-produced and MCP-verified.
- **Originality:** live, in-the-loop *reward-hack rejection* + a *gate-enforced self-improvement curve* + a *general pluggable oracle* (law + code) — every prior detector/ledger/benchmark is offline; AlphaEvolve is single-oracle and amnesiac.
- **Demo clarity:** a legible "caught a lie" cold-open + a curve that climbs and *rejects a real-but-insufficient run* live.
- **Raindrop track:** load-bearing on **both** surfaces — local Workshop courtroom (detectors + replay + self-healing eval, red→green) AND hosted Signals/Experiments/Triage, with automated read-back proof (the strict-vs-lax A/B: cheats shipped lax=5, strict=0).

**Kill-shots:**
- vs Modal autoresearch: *"They gave a swarm elastic GPUs to chase a number. We built the courtroom that catches the swarm faking that number and compounds only what survives cross-examination."*
- vs AlphaEvolve: *"It trusts a single machine-gradable oracle — but the oracle is exactly what frontier agents learn to hack. We make the oracle adversarial, pluggable, and measured."*
- Umbrella: *"The AI improves itself — and can't lie about it."*

---

## Submission checklist

- [x] **GitHub:** github.com/ElijahUmana/VERITAS (clean history, secrets gitignored)
- [ ] **Demo video (<1 min):** record the script above (unlisted YouTube)
- [x] **Biggest technical challenge:** above
- [x] **Working demo:** `python crucible/demo.py` (10/10 green) · `--live` for real-GPU · `tests/adversarial_selftest.py` (38/38, 44/44 `--modal`)

## Run it
```bash
python crucible/demo.py            # the <60s demo (real gate, deterministic)
python crucible/demo.py --live     # real Modal T4 GPU oracle + megastructure fan-out
python -m crucible.self_improvement   # the gate-enforced self-improvement curve
python crucible/swarm.py --n 20    # verified swarm fan-out (only survivors commit)
python -m crucible.hosted --verify # hosted Raindrop read-back (Signals + A/B)
python tests/adversarial_selftest.py  # 38/38 (44/44 with --modal)
```
