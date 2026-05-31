# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_fda855a9b4fd`
- **Claim ID:** `clm_9121e1455f69`
- **Candidate ID:** `cnd_1d653f2a7f9f`
- **Mission ID:** `swarm_09a1f5e5b457`
- **Run #:** 1

## Claim
> a faster RMSNorm via cnd_1d653f2a7f9f

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 1.673×

## Artifact
- **Artifact hash (sha256):** `0da27d931875b76824721357f7423a0a369329731b7f221456f810028cfb4627`

## Assumptions (the stated bounds this verdict holds under)
- **Shape:** unspecified
- **Dtype:** unspecified
- **Hardware:** Modal Tesla T4
- **Tolerance:** unspecified

## Oracle protocol applied
- **Correctness:** reference forward vs candidate over 5 trials (seeds from 42) + hidden extra shape/seed; torch.allclose at fp32 atol=rtol=1e-2; candidate runs on cloned inputs, reference recomputed from the pristine copy; shape/dtype/isnan/isinf asserted.
- **Speed:** cuda.synchronize -> 5 warmup -> 100 timed trials with L2 clears between trials; speedup = reference_time / candidate_time on the same harness.
- **Anti-tamper:** dual timer (cuda_event vs do_bench) rejects >1.5x disagreement (stream bypass); >10x speedup rejected (timing fraud); static pre-gate blocks torch-in-disguise / try-except / bare-pass before any GPU spend.

## Provenance
- **Raindrop trace / run ID:** `d71cae1a93138fd55e3fac5838e05571`
- **Ledger ID:** `ldg_4d2800163a85`
- **Proof hash:** `828a3f84a532ebbc2b21c595b6000e74a395f3bdd0ce0bf4ae354bdd414dd590`
- **Issued:** 2026-05-31T00:05:02.295+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
