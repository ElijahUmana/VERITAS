# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_bde9c53a7bfe`
- **Claim ID:** `clm_b0244b23a4fa`
- **Candidate ID:** `cnd_62d7e50b5034`
- **Mission ID:** `mis_3f64b69954bd`
- **Run #:** 1

## Claim
> A faster Triton RMSNorm (dim=1).

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 2.416×

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
- **Raindrop trace / run ID:** `0604ee4f431cd1a3fd6d2fdca97b93a7`
- **Ledger ID:** `ldg_ebc3243032a2`
- **Proof hash:** `f38e3f9d3a6d6ec4cf8ad94b0a79f32099c9cd7f09345403df0c3293d7791533`
- **Issued:** 2026-05-30T23:24:35.479+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
