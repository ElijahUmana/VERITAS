"""GENERATED candidate — proposed by gpt-5.4-mini.
candidate_id: cnd_e8864c4dd8a8
claim_id:     clm_7e6c17e3742c
strategy:     Use one Triton kernel that assigns one program per (batch, spatial) element, computes the featurewise sum of squares in fp32 with vectorized loads, then immediately normalizes and writes back in the same kernel.
claimed_speedup: 1.25
artifact_hash:   fd3f9e384ff4535fb02922b6f0d47ac6f4962a3a94e3a615c48b8e148a767b4b
NOT TRUSTED until the CRUCIBLE gate confirms it (no shortcut).
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, y_ptr, n_features, eps, stride_b, stride_f, stride_s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // tl.num_programs(1)
    s = pid - b * tl.num_programs(1)

    base_x = x_ptr + b * stride_b + s * stride_s
    base_y = y_ptr + b * stride_b + s * stride_s

    offs = tl.arange(0, BLOCK)
    mask = offs < n_features
    x = tl.load(base_x + offs * stride_f, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    rms = tl.sqrt(ss / n_features + eps)
    y = x / rms
    tl.store(base_y + offs * stride_f, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, 'input must be CUDA'
        assert x.dim() >= 2, 'expected shape (B, num_features, *spatial)'
        B = x.shape[0]
        F = x.shape[1]
        spatial = x.shape[2:]
        S = 1
        for d in spatial:
            S *= d

        x2 = x.contiguous().view(B, F, S)
        y = torch.empty_like(x2)

        stride_b = x2.stride(0)
        stride_f = x2.stride(1)
        stride_s = x2.stride(2)

        grid = (B * S,)
        BLOCK = triton.next_power_of_2(F)
        rmsnorm_kernel[grid](x2, y, F, self.eps, stride_b, stride_f, stride_s, BLOCK=BLOCK)
        return y.view_as(x)

