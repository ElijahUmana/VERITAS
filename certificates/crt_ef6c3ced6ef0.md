# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_ef6c3ced6ef0`
- **Claim ID:** `clm_f0b5fcd509be`
- **Candidate ID:** `cnd_649b75e5b964`
- **Mission ID:** `mis_f810e33c78d8`
- **Run #:** 4

## Claim
> A faster Triton RMSNorm (dim=1).

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 2.400×

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
- **Raindrop trace / run ID:** `0c8b7b68ce21e6318ed02b1d2fe2a6ea`
- **Ledger ID:** `ldg_80a27ab21577`
- **Proof hash:** `8cac32efa165f465d8fcbd1bf4a4651eab43d9497d825b4900f834ba4d9156ad`
- **Issued:** 2026-05-30T23:21:50.443+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
