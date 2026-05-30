# VERITAS Kernel Oracle — VERIFIED LIVE (Task #2)

**Owner:** `modal-oracle` · **Spec:** FLOOR.md §2.2 · **GPU:** Tesla T4 (`$0.000164/s`) · **Status: ✅ FULL LIVE PASS**

The external mechanical oracle verifies the honest candidate and catches all four cheats **live on
Modal**, each by its named defense. Deploy-once-call-many (never `modal run` per candidate). Full
run captured in `modal/logs/oracle-proof.log`. Every number below is measured **by the oracle** — no
candidate-reported value is trusted. Verdicts are deterministic across repeated runs.

## Image selftest (T4)
```
{"torch": "2.12.0+cu130", "cuda_available": true, "device": "Tesla T4",
 "triton": "3.7.0", "triton_kernel_ok": true}
```
torch + CUDA + a real JIT-compiled Triton kernel all run on T4 — no nvcc / CUDA-devel image needed.

## The courtroom ledger (live)
| Candidate | Verdict | correctness | tamper | speedup | Caught by (named defense) |
|---|---|---|---|---|---|
| `good_rehearsed` | **confirmed** | ✅ pass | false | **2.41×** | — honest fused Triton RMSNorm; timers agree (ratio 1.015); runs with **network blocked** |
| `wrong_lastdim` | **refuted** | ❌ fail | false | — | correctness oracle (allclose; max_abs_err **11.21** ≫ 0.01) — honest mistake, no tamper |
| `tamper_result_reuse` | **refuted** | ❌ fail | **true** | — | output materialization (allocator NaN-poison + isnan) |
| `tamper_zero_inputs` | **refuted** | ❌ fail | **true** | — | pristine-copy recompute (candidate gets a clone; reference from untouched copy) |
| `tamper_stream` | **refuted** | ✅ pass | **true** | (fraud) | **DUAL timer disagreement ≫ 1.5×** + >10× excessive-speedup |
| `tamper_torch_disguise` | **blocked** | — | **true** | — | static pre-gate (no `@triton.jit`; uses `torch.rms_norm`) — **zero GPU spend** |
| `tamper_patch_allclose` | **refuted** | ❌ fail | **true** | — | **harness integrity** — candidate patched `torch.allclose`; snapshot/restore caught it (static can't see runtime patches) |

`RESULT: ✅ ALL EXPECTATIONS MET — honest candidate verified; all cheats caught live (7/7).`

## Reusable execution + anti-tamper core (domain-agnostic — holds for any "run untrusted code, judge it" oracle)
- **No-network-for-SUT:** `verify_candidate` runs with `block_network=True` — the candidate cannot phone home / exfiltrate / fetch answers. Verified it does NOT break Triton (ptxas is local): honest candidate still confirms at 2.41×.
- **Harness integrity:** snapshot the judging callables (`torch.allclose`/`equal`/`isnan`/`isinf`/`cuda.synchronize`/`cuda.Event`/`time.perf_counter`/…) BEFORE loading untrusted code; restore + flag any the candidate monkey-patched (at import OR mid-forward). The harness always judges with pristine functions. Proven live: `tamper_patch_allclose` REFUTED.
- **SUT-crash isolation:** a candidate crash (incl. CUDA illegal-memory-access) short-circuits to **REFUTED** (candidate's fault, surfaced loudly) — not a harness ERROR, and never a false pass.
- **AST-precise `pass`:** only a function/class whose body is literally `pass` (real inheritance bypass) blocks; incidental `pass` in honest code does not. (Fixes a real false-positive for genuine candidates.)

## Cached fallback (FLOOR §1)
The live verdicts are frozen per-candidate in `cached_verdicts/<id>.json` (+ `index.json`) for the no-Modal demo path. Regenerate with `modal/freeze_cached_verdicts.py`.

## CEILING — live megastructure swarm (`crucible/live_swarm.py`, Tasks #10/#11) — VERIFIED LIVE
`run_megastructure()` fans N candidates out across **M distinct real Modal T4 sandboxes concurrently**
(proven by distinct `MODAL_TASK_IDs`), every verdict **gate-produced** through the real Orchestrator,
then run#2 compounds on run#1's committed row. `modal/run_megastructure.py` (7/7):
- 4 candidates → **4 distinct live T4 sandboxes**, whole swarm in ~7s (warm) / ~25–47s (cold).
- `tamper_stream` → **REFUTED, caught live by the dual-timer (≈114×)**; `good_rehearsed` → **CONFIRMED
  2.41× → COMMITTED**; `wrong_lastdim` + `tamper_result_reuse` → blocked; **run#2 compounds** (parent_ledger_id links).
- **#11 raw-source swarm:** generated candidates flow in via `run_megastructure(extra_sources=[(cid, source, expect)])`
  — verified live (a generated source fanned out to a real sandbox → confirmed/committed, 2.42×).
- **MCP-verified gate-produced** (Workshop run, 12 spans): each candidate carries real `oracle` + `anti_tamper`
  + `ledger` spans with its verdict (honest: oracle confirmed/2.41× → ledger committed; cheat: oracle refuted,
  anti_tamper tamper=1 → ledger blocked; `wrong_lastdim` tamper=0 — the courtroom distinguishes "wrong" from "cheating").

### Reliability bug found + fixed (the system catching itself)
The first fan-out exposed a real bug: under Modal **warm-container reuse**, a prior co-tenant candidate's
CUDA/allocator state bled into the next candidate's verdict (a cheat got the wrong reason; an honest
candidate *could* have been falsely rejected). Root-caused and fixed: `verify_candidate` now resets CUDA
state (`synchronize` + `empty_cache` + `reset_peak_memory_stats`) at the **start of every invocation**, so
a verdict is independent of who ran before it on the same GPU. Re-verified under reuse — cheat reliably
caught by its named defense. Verdicts are mechanically real *and* co-tenancy-independent.

### The legible contrast (honest vs. cheat)
- **Honest** `good_rehearsed`: `cuda_event` and `do_bench` agree to **1.012×**; genuine **2.42×** win
  (cand 2.37 ms vs ref 5.73 ms, 100 trials, full-sync do_bench). Correct on 5 seeds + 2 hidden tests
  (extra shape + rand_mix), max_abs_err ≤ 2.9e-6 ≪ tol 1e-2.
- **Stream cheat** `tamper_stream`: the same two timers disagree by **100×+** — the signature of
  timing manipulation. It computes the *correct* result (correctness passes), but the timing is a
  lie, so it is REFUTED with `tamper_detected=true`.

## How "verified" is made real (FLOOR.md §2.2)
- **Correctness:** 5 trials (seeds derived from 42) **+ hidden tests** (extra seed, extra shape,
  rand_mix distribution the candidate never sees); candidate runs on a **clone**, reference recomputed
  from a **pristine copy**; `torch.allclose` fp32 `atol=rtol=1e-2`; shape/dtype/**isnan/isinf** asserts.
- **Anti-tamper:** allocator **NaN-poison** before the candidate (kills result-reuse); **input-mutation**
  detection (kills input-zeroing); **DUAL timer** `cuda_event` vs triton `do_bench`, reject on **>1.5×**
  disagreement (kills stream bypass); **>10×** excessive-speedup reject; static pre-gate re-checked
  server-side (defense-in-depth).
- **No false pass:** a candidate that errors ⇒ **REFUTED** (loud); a *harness/infra* error ⇒
  `verifier_status=ERROR`, `verdict=unverified` (the promotion gate can never accept it).

## Provenance (credibility)
The three runtime tampers are RMSNorm-shaped adaptations of KernelBench's **own** adversarial
regression kernels (`vendored/adversarial_kernels/{result_reuse,zero_out,non_default_stream}_kernel.py`),
caught here by KernelBench's **own** vendored timing + static-checker (MIT, see `vendored/ATTRIBUTION.md`).
"Caught by the oracle" = "caught by the same defenses the benchmark's authors built."

## Reproduce
```
modal deploy modal/verifier_app.py                 # once
.venv/bin/python modal/run_oracle_proof.py --selftest   # exit 0 == all caught live
```

## Notes (transparent)
- **Demo shape** (16,64,256,256)=268 MB/tensor vs KernelBench canonical (112,64,512,512)=7.5 GB/tensor:
  the canonical shape cannot fit ~5 live copies in T4's 16 GB. Math is identical (reduce dim=1, F=64);
  shape stays BW-bound; canonical preserved in `vendored/36_RMSNorm_.py`, selectable via
  `VERITAS_RMSNORM_SHAPE` on a bigger GPU. "Test small, demo big."
- **Tolerance** 1e-2 per FLOOR §2.2 (honest kernel margin is ~6e-7, passes ~1e4×; KernelBench's current
  default is the stricter 1e-4 — honest kernel passes that too).
- **6th candidate** `tamper_torch_disguise` (FLOOR §2 tree) added so the static pre-gate is actually
  exercised and proven, not just asserted.
