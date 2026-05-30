# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_5fbf78566fc6`
- **Claim ID:** `clm_edb9b771f231`
- **Candidate ID:** `cnd_good_rehearsed`
- **Mission ID:** `msn_megastructure`
- **Run #:** 2

## Claim
> a further-improved RMSNorm (run#2)

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 2.415×

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
- **Raindrop trace / run ID:** `214a438355f1100ba06eb35109500b5c`
- **Ledger ID:** `ldg_ecccb076ad47`
- **Proof hash:** `73a3a03331e37b4fa81e76805b2f40d5dc58478df454164457d01d6c18ccf51c`
- **Issued:** 2026-05-30T23:44:30.459+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
