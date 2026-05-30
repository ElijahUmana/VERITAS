"""GENERATED candidate — proposed by gpt-5.4-mini.
candidate_id: cnd_3f3ff470961c
claim_id:     clm_40c1128051b6
strategy:     One-pass Triton RMSNorm over the feature axis with per-spatial-element blocks, using vectorized loads and a fused reduction + normalization kernel.
claimed_speedup: 1.2
artifact_hash:   1c1ee0a4239a016ebe19f599d48cb88e609953ab266e9669b5c24e557784c3b0
NOT TRUSTED until the CRUCIBLE gate confirms it (no shortcut).
"""
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_dim1_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
    numel: tl.constexpr,
    BLOCK_S: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    s_offsets = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # For each spatial position, reduce over C.
    # x is laid out as [B, C, S] (contiguous assumption after flattening trailing dims).
    # Generic flattened spatial handling: index = (b * C + c) * S + s
    b = s_offsets // (S // B)  # placeholder-style safety not used directly below

    # We compute each spatial element independently, mapping linearized spatial index.
    for s in range(0, BLOCK_S):
        pass


@triton.jit
def rmsnorm_dim1_kernel_2d(
    x_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
    stride_b: tl.constexpr, stride_c: tl.constexpr, stride_s: tl.constexpr,
    BLOCK_S: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s < S

    # Flattened index over spatial positions; for each s, reduce over batch-feature axis.
    # Here x is assumed contiguous in [B, C, S] after reshaping trailing dims into S.
    # We support generic contiguous tensors by using provided strides.
    x2_sum = tl.zeros([BLOCK_S], dtype=tl.float32)
    for c in range(0, C):
        x = tl.load(x_ptr + c * stride_c + s * stride_s, mask=mask_s, other=0.0)
        x2_sum += x * x
    rms = tl.sqrt(x2_sum / C + EPS)
    inv = 1.0 / rms
    for c in range(0, C):
        x = tl.load(x_ptr + c * stride_c + s * stride_s, mask=mask_s, other=0.0)
        tl.store(y_ptr + c * stride_c + s * stride_s, x * inv, mask=mask_s)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce over dim=1, preserve remaining dims.
        if not x.is_cuda:
            rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
            return x / rms

        B = x.shape[0]
        C = x.shape[1]
        spatial_shape = x.shape[2:]
        S = 1
        for d in spatial_shape:
            S *= d

        x_ = x.contiguous().view(B, C, S)
        y = torch.empty_like(x_)

        # One program per spatial block.
        BLOCK_S = 128
        grid = (triton.cdiv(S, BLOCK_S),)
        rmsnorm_dim1_kernel_2d[grid](
            x_, y,
            B=B, C=C, S=S,
            stride_b=x_.stride(0), stride_c=x_.stride(1), stride_s=x_.stride(2),
            BLOCK_S=BLOCK_S,
            EPS=self.eps,
            num_warps=4,
        )
        return y.view_as(x)

