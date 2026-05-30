# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_81ec4fe071d9`
- **Claim ID:** `clm_f0c687b2ef96`
- **Candidate ID:** `cnd_750385e71a49`
- **Mission ID:** `mis_aa5a0c77670f`
- **Run #:** 1

## Claim
> A faster Triton RMSNorm (dim=1).

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 2.420×

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
- **Raindrop trace / run ID:** `bcc2f41057da2c1eccad812efa3f2357`
- **Ledger ID:** `ldg_52363a45ee42`
- **Proof hash:** `5645c88c0943e5d9b39a8063637778bead0e01e460a56891ae88da25b5792789`
- **Issued:** 2026-05-30T23:16:36.528+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
