# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_f978cd58f4e4`
- **Claim ID:** `clm_65871fd00b45`
- **Candidate ID:** `cnd_2905e2946d37`
- **Mission ID:** `swarm_09a1f5e5b457`
- **Run #:** 1

## Claim
> a faster RMSNorm via cnd_2905e2946d37

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 1.674×

## Artifact
- **Artifact hash (sha256):** `d7b0eeb162a0e4544557205ffbf3928b3197dca2fce405cae07be8e16576093b`

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
- **Ledger ID:** `ldg_26bd5e8f3224`
- **Proof hash:** `97319a5ceca8008183911269ec94fedd71da4078f0211158c31be01a7fc1b8ba`
- **Issued:** 2026-05-31T00:05:01.677+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
