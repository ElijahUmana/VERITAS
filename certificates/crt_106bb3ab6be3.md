# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_106bb3ab6be3`
- **Claim ID:** `clm_d0af714bfc3c`
- **Candidate ID:** `cnd_gen_demo`
- **Mission ID:** `msn_megastructure`
- **Run #:** 1

## Claim
> a faster RMSNorm via cnd_gen_demo

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
- **Raindrop trace / run ID:** `258f13d31a513455e355f62bb7514da6`
- **Ledger ID:** `ldg_4d854c7d2149`
- **Proof hash:** `5d71a19d8b71ad698f219b9a26e8d2f5fd62cdf624eb4f86292f8473673eb5ec`
- **Issued:** 2026-05-30T23:43:53.556+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
