"""GENERATED candidate — proposed by gpt-5.4-mini.
candidate_id: cnd_d62d51603f2e
claim_id:     clm_909cccbf3c64
strategy:     Fuse RMS accumulation, normalization, and writeback in one Triton kernel with vectorized per-spatial-element blocks over the feature axis.
claimed_speedup: 1.4
artifact_hash:   dd323d68624476b23fed986d455849371177530002fb44846090a7d28530af31
NOT TRUSTED until the CRUCIBLE gate confirms it (no shortcut).
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, y_ptr,
    stride_b, stride_f, stride_s,
    num_features: tl.constexpr,
    eps: tl.constexpr,
    S: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid

    # x/y layout: [B, F, S] contiguous on the last dimension after flattening spatial dims
    # We process one spatial position per program.
    offs_f = tl.arange(0, BLOCK_F)
    f_mask = offs_f < num_features

    # Compute sum of squares in fp32 over feature axis.
    acc = tl.zeros([BLOCK_F], dtype=tl.float32)
    x_base = x_ptr + s * stride_s + offs_f * stride_f
    x = tl.load(x_base, mask=f_mask, other=0.0)
    acc += x * x
    sumsq = tl.sum(acc, axis=0)
    mean_sq = sumsq / num_features
    inv_rms = tl.math.rsqrt(mean_sq + eps)

    # Normalize and store.
    y = x * inv_rms
    y_ptrs = y_ptr + s * stride_s + offs_f * stride_f
    tl.store(y_ptrs, y, mask=f_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        assert x.ndim >= 2
        B = x.shape[0]
        F = x.shape[1]
        spatial = orig_shape[2:]
        S = 1
        for d in spatial:
            S *= d
        x3 = x.reshape(B, F, S).contiguous()
        y3 = torch.empty_like(x3)

        grid = (B * S,)
        # One program per (batch, spatial) position.
        rmsnorm_fwd_kernel[grid](
            x3, y3,
            x3.stride(0), x3.stride(1), x3.stride(2),
            num_features=F,
            eps=self.eps,
            S=S,
            BLOCK_F=triton.next_power_of_2(F),
        )
        return y3.reshape(orig_shape)

