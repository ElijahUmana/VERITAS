# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_6df377cab32a`
- **Claim ID:** `clm_632433924807`
- **Candidate ID:** `cnd_973cf3285258`
- **Mission ID:** `swarm_7597b0e29399`
- **Run #:** 1

## Claim
> a faster RMSNorm via cnd_973cf3285258

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 1.625×

## Artifact
- **Artifact hash (sha256):** `3dfcf00a1ef6c5b91c4916beb792f9549678cfec91edbb336e8f190914a85820`

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
- **Raindrop trace / run ID:** `0d41d18e57f74ebb66fc1e11d04bb523`
- **Ledger ID:** `ldg_82da475d193f`
- **Proof hash:** `42b3c6a2bdd93be970e50ab2e56d3a4cf5a426850a79e21bcf0dd5f5d775c357`
- **Issued:** 2026-05-31T00:01:51.389+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
