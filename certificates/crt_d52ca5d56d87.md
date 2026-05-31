# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_d52ca5d56d87`
- **Claim ID:** `clm_6c1c832b1b78`
- **Candidate ID:** `cnd_6f16c922cf84`
- **Mission ID:** `swarm_09a1f5e5b457`
- **Run #:** 1

## Claim
> a faster RMSNorm via cnd_6f16c922cf84

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 1.722×

## Artifact
- **Artifact hash (sha256):** `5181f8dfae49ccc74a02ba5918f17958c34570e9d5c8a4d5914a45f216ede1c0`

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
- **Ledger ID:** `ldg_78197092107b`
- **Proof hash:** `e7b5bb18a7308efc247497df7d2fd67876fbce0446d5090b37cb99a260f03e1b`
- **Issued:** 2026-05-31T00:05:02.070+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
