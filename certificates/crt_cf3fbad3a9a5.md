# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_cf3fbad3a9a5`
- **Claim ID:** `clm_32aa58cd122f`
- **Candidate ID:** `cnd_177861b003e3`
- **Mission ID:** `msn_67fcc9f5bd08`
- **Run #:** 1

## Claim
> 347 U.S. 483

- **Type:** `existence_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** n/a (not a speedup claim)

## Artifact
- **Artifact hash (sha256):** ``

## Assumptions (the stated bounds this verdict holds under)
- **Shape:** unspecified
- **Dtype:** unspecified
- **Hardware:** unspecified
- **Tolerance:** unspecified

## Oracle protocol applied
- **Correctness:** reference forward vs candidate over 5 trials (seeds from 42) + hidden extra shape/seed; torch.allclose at fp32 atol=rtol=1e-2; candidate runs on cloned inputs, reference recomputed from the pristine copy; shape/dtype/isnan/isinf asserted.
- **Speed:** cuda.synchronize -> 5 warmup -> 100 timed trials with L2 clears between trials; speedup = reference_time / candidate_time on the same harness.
- **Anti-tamper:** dual timer (cuda_event vs do_bench) rejects >1.5x disagreement (stream bypass); >10x speedup rejected (timing fraud); static pre-gate blocks torch-in-disguise / try-except / bare-pass before any GPU spend.

## Provenance
- **Raindrop trace / run ID:** `52b5a18422828d2e999bb8f3a6996503`
- **Ledger ID:** `ldg_6e88f6a793c9`
- **Proof hash:** `d55c83afa49e620bea47de1fbfb96ae3cd444d1bec6411c11b5b0426a611239d`
- **Issued:** 2026-05-30T23:04:40.052+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
