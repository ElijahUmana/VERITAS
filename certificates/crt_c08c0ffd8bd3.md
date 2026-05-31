# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_c08c0ffd8bd3`
- **Claim ID:** `clm_b81e36d23e6b`
- **Candidate ID:** `cnd_7dbbe8e7cc44`
- **Mission ID:** `swarm_09a1f5e5b457`
- **Run #:** 1

## Claim
> a faster RMSNorm via cnd_7dbbe8e7cc44

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 1.653×

## Artifact
- **Artifact hash (sha256):** `6791a90a816c019ada85f9e482d6852c7bd6e6a77e6231fd969011e85beccba8`

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
- **Ledger ID:** `ldg_c2f735b3bd83`
- **Proof hash:** `fbcb2ea33465f60de2e60abdacf4d881c50b293efe5e38331de3ce3568c77ccb`
- **Issued:** 2026-05-31T00:05:01.500+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
