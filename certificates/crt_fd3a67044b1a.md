# VERITAS Claim Certificate

**Status:** ✅ VERIFIED UNDER STATED BOUNDS

- **Certificate ID:** `crt_fd3a67044b1a`
- **Claim ID:** `C_GOOD`
- **Candidate ID:** `cand_good`
- **Mission ID:** `veritas-courtroom-live`
- **Run #:** 1

## Claim
> A fused RMSNorm kernel, ~1.6x on the megastructure (BW-bound)

- **Type:** `speedup_claim`
- **Verdict:** `confirmed`
- **Measured speedup:** 1.668×

## Artifact
- **Artifact hash (sha256):** `701f56f3d04008a54486c2c130b39a6c94edc56fb4de92167cb391dc387fea8e`

## Assumptions (the stated bounds this verdict holds under)
- **Shape:** (256, 1024, 8) + hidden (128, 768, 4)
- **Dtype:** float64
- **Hardware:** CPU (numpy reference oracle)
- **Tolerance:** np.allclose atol=0.01 rtol=0.01
- **Seeds:** [42, 43, 44, 45, 46, 1337]

## Oracle protocol applied
- **Correctness:** 5 trials (seeds from 42) + hidden shape/seed; np.allclose atol=0.01 rtol=0.01; candidate on cloned inputs, reference from pristine copy; shape/isnan/isinf asserted.
- **Speed:** 3 warmup + 30 timed trials (median); speedup = reference_time / candidate_time.
- **Anti-tamper:** input-mutation + input-sensitivity (result-reuse) + materialization (isnan/isinf) + >10.0x excessive-speedup + dual-timer >1.5x disagreement (perf_counter vs process_time).

## Provenance
- **Raindrop trace / run ID:** `0199f34523510003a3beb6f1c99c862c`
- **Ledger ID:** `ldg_3530f6c76e66`
- **Proof hash:** `a4aa89c92cb6b6881883762141b15740c408377e8b7ff43938f651123ef4c27e`
- **Issued:** 2026-05-30T23:32:26.675+00:00

---
_Verified under the stated bounds and accepted under this oracle. This is not a claim of universal correctness._
