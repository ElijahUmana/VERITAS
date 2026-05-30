"""VERITAS rmsnorm_lab — the reference oracle target: KernelBench level1/36_RMSNorm_.

The reference `Model.forward` is the **unmodified** KernelBench reference (vendored verbatim in
`vendored/36_RMSNorm_.py`):

    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)   # reduce dim=1, eps INSIDE sqrt
    return x / rms                                                          # no learnable weight

The correctness oracle compares a candidate's `ModelNew` against this `Model` exactly.

This module is intentionally **import-safe without torch** (it is mainly a *source emitter*: the
authoritative reference is the `reference_source()` string the Modal worker execs). torch is
imported lazily only by the runtime helpers, so the client/oracle can build the source string
on a machine that has no torch installed.

INPUT SHAPE — a deliberate, flagged choice (not a silent spec change):
  KernelBench's canonical shape is (112, 64, 512, 512) fp32 = **7.5 GB per tensor**. The oracle
  must hold input + pristine copy + candidate clone + reference output + candidate output at
  once (~5 live tensors ≈ 37 GB) — it cannot run on the T4 (16 GB) demo GPU. So the DEMO shape
  is (16, 64, 256, 256) fp32 = 268 MB/tensor (~1.6 GB working set), which keeps identical math
  (reduce over dim=1, features=64) and stays far larger than the T4 L2 (4 MB) so the problem is
  **bandwidth-bound** and the honest fused Triton kernel shows a genuine speedup. The verbatim
  7.5 GB shape is preserved in `vendored/36_RMSNorm_.py` and selectable via VERITAS_RMSNORM_SHAPE
  on a larger GPU. "Test small, demo big" — same mechanism, sized to the GPU.
"""
from __future__ import annotations

import os

# Canonical KernelBench shape (preserved): batch=112, features=64, dim1=512, dim2=512 (7.5 GB/tensor).
CANONICAL_SHAPE = (112, 64, 512, 512)
# T4 demo shape: 268 MB/tensor, bandwidth-bound, fits 16 GB with all live copies.
DEMO_SHAPE = (16, 64, 256, 256)
FEATURES = 64  # init arg; the reduction dim (dim=1) must equal this


def shape() -> tuple[int, int, int, int]:
    """Resolve the demo input shape (override via VERITAS_RMSNORM_SHAPE='B,F,D1,D2')."""
    raw = os.environ.get("VERITAS_RMSNORM_SHAPE", "").strip()
    if not raw:
        return DEMO_SHAPE
    parts = tuple(int(p) for p in raw.replace("x", ",").split(","))
    assert len(parts) == 4, f"VERITAS_RMSNORM_SHAPE must be B,F,D1,D2 — got {raw!r}"
    assert parts[1] == FEATURES, f"features (dim=1) must be {FEATURES}, got {parts[1]}"
    return parts  # type: ignore[return-value]


# The unmodified KernelBench 36_RMSNorm_ math, emitted as a self-contained source string.
_REFERENCE_SRC_TEMPLATE = '''\
import torch
import torch.nn as nn


class Model(nn.Module):
    """RMS Normalization — KernelBench level1/36_RMSNorm_ reference (unmodified)."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return x / rms


_B, _F, _D1, _D2 = {shape}


def get_inputs():
    return [torch.rand(_B, _F, _D1, _D2)]


def get_init_inputs():
    return [_F]
'''


def reference_source() -> str:
    """Self-contained reference source string fed to the oracle as `reference_src`."""
    return _REFERENCE_SRC_TEMPLATE.format(shape=tuple(shape()))


# ----- Runtime convenience (torch required; used inside the worker / when torch is present) -----
def get_inputs():
    import torch  # lazy

    b, f, d1, d2 = shape()
    return [torch.rand(b, f, d1, d2)]


def get_init_inputs():
    return [FEATURES]


def load_model():
    """exec the reference source and return (Model, get_init_inputs, get_inputs). Needs torch."""
    ctx: dict = {}
    exec(reference_source(), ctx)  # noqa: S102 — trusted fixed source
    return ctx["Model"], ctx["get_init_inputs"], ctx["get_inputs"]


if __name__ == "__main__":
    print(reference_source())
