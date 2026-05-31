# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_18a34ec94a9f`
- **Claim ID:** `clm_5739bfd708c3`
- **Candidate ID:** `cnd_3de8cf635e61`
- **Mission ID:** `swarm_7597b0e29399`
- **Run #:** 1

## Claim
> a faster RMSNorm via cnd_3de8cf635e61

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 2.372×

## Artifact
- **Artifact hash (sha256):** `a7a795d9695a4c8c2728bb82bd740d654e4277211c32e451d63132a8b440f840`

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
- **Raindrop trace / run ID:** `0d41d18e57f74ebb66fc1e11d04bb523`
- **Ledger ID:** `ldg_fe6eb324dd99`
- **Proof hash:** `87f8170bbcbd8e0fd7077af4d258e47ea78f5e8ce3f84456eae911f87326f115`
- **Issued:** 2026-05-31T00:01:51.483+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
