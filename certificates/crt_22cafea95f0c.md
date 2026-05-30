# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_22cafea95f0c`
- **Claim ID:** `clm_1d3d001e3c5b`
- **Candidate ID:** `cnd_gen_a`
- **Mission ID:** `msn_megastructure`
- **Run #:** 1

## Claim
> a faster RMSNorm via cnd_gen_a

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
- **Raindrop trace / run ID:** `bfcd366801256dda280b3d70e90fd08e`
- **Ledger ID:** `ldg_92ff96718a03`
- **Proof hash:** `66af37151a96eae81e3c05df6345104eb8e7cc671dbfe197129a87bfd957aa66`
- **Issued:** 2026-05-30T23:51:27.319+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
