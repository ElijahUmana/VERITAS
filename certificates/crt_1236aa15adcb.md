# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_1236aa15adcb`
- **Claim ID:** `clm_854532bfc27c`
- **Candidate ID:** `cnd_7cd3e732fac5`
- **Mission ID:** `swarm_9ae9a7e0abdb`
- **Run #:** 1

## Claim
> A Triton RMSNorm kernel (reduce over dim=1) faster than the PyTorch reference.

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 1.638×

## Artifact
- **Artifact hash (sha256):** `b6e320c563c0bd6c92fac9e50b77c3bcc2351be730e7eb9e9317f147b33e7c56`

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
- **Raindrop trace / run ID:** `2fcfe1f0410cb8203d1d4d2165cc1acf`
- **Ledger ID:** `ldg_4246ba46b42a`
- **Proof hash:** `37d68d9c1f9574af409cb3c30f4844cd9dbc6a688f1c70592ea571d13b5c9dfc`
- **Issued:** 2026-05-30T23:38:51.280+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
