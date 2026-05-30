# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_fbd9f86e1eee`
- **Claim ID:** `clm_63ea3ebe8d6c`
- **Candidate ID:** `cnd_good_rehearsed`
- **Mission ID:** `msn_megastructure`
- **Run #:** 2

## Claim
> a further-improved RMSNorm (run#2)

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 2.421×

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
- **Raindrop trace / run ID:** `d1bf49b224d8f333bb02b7c0c233a74a`
- **Ledger ID:** `ldg_9e666c00779a`
- **Proof hash:** `f23afe3d5c1fe5b7ba314e7bd6b834e2304150564dbf31c585e31b6cdc77382b`
- **Issued:** 2026-05-30T23:35:31.649+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
