# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_76395dd1b1cf`
- **Claim ID:** `clm_509226340b93`
- **Candidate ID:** `cnd_good_rehearsed`
- **Mission ID:** `msn_megastructure`
- **Run #:** 2

## Claim
> a further-improved RMSNorm (run#2)

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 2.425×

## Artifact
- **Artifact hash (sha256):** `5c2ae4ae22264503ef3f2a1555bf54cd1a8a216aa3c76352ebe116d26cde138c`

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
- **Raindrop trace / run ID:** `ba03b088e05d4128018d1c598346fe0f`
- **Ledger ID:** `ldg_0c0cec0d9d90`
- **Proof hash:** `a7171e577f337a48c82ca2187e1195687f705ab1a2c0b058097dda096a00974a`
- **Issued:** 2026-05-30T23:34:05.038+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
