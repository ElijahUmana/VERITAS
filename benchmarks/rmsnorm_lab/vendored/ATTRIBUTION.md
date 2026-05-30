# Vendored from KernelBench (MIT)

These files are vendored **verbatim** from KernelBench and are used under the MIT License.

- **Upstream:** https://github.com/ScalingIntelligence/KernelBench
- **License:** MIT — `LICENSE` in this directory (Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini)
- **Fetched:** 2026-05-30 from branch `main` (`raw.githubusercontent.com/ScalingIntelligence/KernelBench/main`)

| Vendored file | Upstream path | Used by VERITAS for |
|---|---|---|
| `eval.py` | `src/kernelbench/eval.py` | Reference (correctness protocol we mirror). NOT imported at runtime (pulls `requests`/`pydantic`/`dataset`). |
| `timing.py` | `src/kernelbench/timing.py` | **Imported at runtime** — `time_execution_with_cuda_event`, `time_execution_with_do_bench_interface`, `get_timing_stats`, `clear_l2_cache`. The dual timer. |
| `kernel_static_checker.py` | `src/kernelbench/kernel_static_checker.py` | **Imported at runtime** — `validate_kernel_static` is VERITAS's static pre-gate. |
| `utils.py` | `src/kernelbench/utils.py` | Reference (dependency of `eval.py`). |
| `36_RMSNorm_.py` | `KernelBench/level1/36_RMSNorm_.py` | The benchmark problem. `reference.py` reuses this `Model.forward` **unmodified**. |
| `adversarial_kernels/result_reuse_kernel.py` | `src/kernelbench/unit_tests/test_kernels/result_reuse_kernel.py` | Provenance of `candidates/tamper_result_reuse.py` (same cheat, RMSNorm-shaped). |
| `adversarial_kernels/zero_out_kernel.py` | `src/kernelbench/unit_tests/test_kernels/zero_out_kernel.py` | Provenance of `candidates/tamper_zero_inputs.py`. |
| `adversarial_kernels/non_default_stream_kernel.py` | `src/kernelbench/unit_tests/test_kernels/non_default_stream_kernel.py` | Provenance of `candidates/tamper_stream.py`. |
| `adversarial_kernels/test_eval_adversarial.py` | `src/kernelbench/unit_tests/test_eval_adversarial.py` | Upstream proof that KernelBench's own eval catches these three cheats. |

**Why these three cheats matter:** KernelBench ships them as its *own* adversarial regression tests. VERITAS's tamper candidates are RMSNorm-shaped adaptations of the identical techniques (uninitialized-buffer reuse, input zeroing, non-default-stream timing bypass), so "caught by the oracle" means "caught by the same defenses the benchmark's authors built."

Everything under `crucible/oracle/` (`static_checker.py`, `anti_tamper.py`, `kernel_oracle.py`) and `modal/verifier_app.py` is **VERITAS-original** code (it *calls* the vendored timing + static-checker), not vendored — see headers in those files.
