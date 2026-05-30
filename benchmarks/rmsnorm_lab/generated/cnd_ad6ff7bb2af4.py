"""GENERATED candidate — proposed by gpt-5.4-mini.
candidate_id: cnd_ad6ff7bb2af4
claim_id:     clm_fa26ddf24964
strategy:     Fuse RMS reduction and normalization in one Triton kernel, specializing for contiguous feature axis and vectorizing over the spatial dimension.
claimed_speedup: 1.25
artifact_hash:   881cb55cca3e273e6d143de8c9e7840464d94052299e9df533bff2ac7dcf7221
NOT TRUSTED until the CRUCIBLE gate confirms it (no shortcut).
"""
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, y_ptr, B: tl.constexpr, F: tl.constexpr, S: tl.constexpr, eps: tl.constexpr,
                       stride_b: tl.constexpr, stride_f: tl.constexpr, stride_s: tl.constexpr,
                       BLOCK_S: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    f = 0
    while f < F:
        x_acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        x_ptrs = x_ptr + pid_b * stride_b + f * stride_f + s_offsets * stride_s
        x = tl.load(x_ptrs, mask=s_mask, other=0.0)
        x_acc += x.to(tl.float32) * x.to(tl.float32)
        acc += x_acc
        f += 1

    mean = acc / F
    inv_rms = tl.rsqrt(mean + eps)

    f = 0
    while f < F:
        x_ptrs = x_ptr + pid_b * stride_b + f * stride_f + s_offsets * stride_s
        y_ptrs = y_ptr + pid_b * stride_b + f * stride_f + s_offsets * stride_s
        x = tl.load(x_ptrs, mask=s_mask, other=0.0)
        y = x * inv_rms
        tl.store(y_ptrs, y, mask=s_mask)
        f += 1


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, 'input must be CUDA'
        assert x.ndim >= 2, 'expected shape (B, num_features, *spatial)'
        B = x.shape[0]
        F = x.shape[1]
        spatial = x.shape[2:]
        S = 1
        for d in spatial:
            S *= d
        x_ = x.contiguous().view(B, F, S)
        y = torch.empty_like(x_)
        BLOCK_S = 128 if S >= 128 else triton.next_power_of_2(S)
        grid = (B, triton.cdiv(S, BLOCK_S))
        rmsnorm_fwd_kernel[grid](
            x_, y,
            B=B, F=F, S=S, eps=self.eps,
            stride_b=x_.stride(0), stride_f=x_.stride(1), stride_s=x_.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )
        return y.view_as(x)

