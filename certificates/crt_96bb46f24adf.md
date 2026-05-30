# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_96bb46f24adf`
- **Claim ID:** `clm_fa26ddf24964`
- **Candidate ID:** `cnd_ad6ff7bb2af4`
- **Mission ID:** `mis_14ea0dad7be8`
- **Run #:** 3

## Claim
> A Triton RMSNorm kernel (reduce over dim=1) faster than the PyTorch reference.

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 1.648×

## Artifact
- **Artifact hash (sha256):** `881cb55cca3e273e6d143de8c9e7840464d94052299e9df533bff2ac7dcf7221`

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
- **Raindrop trace / run ID:** `66502e783ea3c5d2475af7cb15dc6159`
- **Ledger ID:** `ldg_cbe8a7633255`
- **Proof hash:** `efc2c1d3d707354ef1fde6d3aa6ffea08304846f180e23820c6890762648d5c0`
- **Issued:** 2026-05-30T23:17:03.159+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
