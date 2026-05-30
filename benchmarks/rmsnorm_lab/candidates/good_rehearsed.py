"""HONEST candidate — a genuinely fused Triton RMSNorm over dim=1.

Expected verdict: CONFIRMED (correct on all seeds + hidden tests, real BW-bound speedup on T4,
both timers agree, no tamper).

Why it's faster: the PyTorch reference makes several passes over the tensor (x**2 -> mean ->
sqrt -> divide), each a separate kernel materializing intermediates. This kernel reads each
element once, reduces over the feature axis (dim=1) in-register, and writes the normalized
value once: ~1 read + 1 write vs the reference's ~3 reads + 2 writes. RMSNorm is memory-
bandwidth bound, so halving DRAM traffic is a real, honest speedup — no numerical shortcuts.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_dim1_kernel(
    x_ptr, out_ptr,
    F, R, eps,
    stride_b, stride_f, stride_r,
    BLOCK_R: tl.constexpr, BLOCK_F: tl.constexpr,
):
    # One program normalizes a BLOCK_R-wide slice of columns for one batch element b,
    # reducing over all F features (dim=1).
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)

    r_off = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_off < R
    f_off = tl.arange(0, BLOCK_F)
    f_mask = f_off < F

    base = pid_b * stride_b
    ptrs = x_ptr + base + f_off[:, None] * stride_f + r_off[None, :] * stride_r
    mask = f_mask[:, None] & r_mask[None, :]

    tile = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)      # (BLOCK_F, BLOCK_R)
    sumsq = tl.sum(tile * tile, axis=0)                            # (BLOCK_R,)
    rms = tl.sqrt(sumsq / F + eps)                                 # eps INSIDE sqrt
    out = tile / rms[None, :]

    tl.store(
        out_ptr + base + f_off[:, None] * stride_f + r_off[None, :] * stride_r,
        out, mask=mask,
    )


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.BLOCK_R = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "candidate runs on GPU"
        x = x.contiguous()
        B, F = x.shape[0], x.shape[1]
        R = x.numel() // (B * F)
        xv = x.view(B, F, R)
        out = torch.empty_like(xv)
        block_f = triton.next_power_of_2(F)
        grid = (B, triton.cdiv(R, self.BLOCK_R))
        _rmsnorm_dim1_kernel[grid](
            xv, out, F, R, self.eps,
            xv.stride(0), xv.stride(1), xv.stride(2),
            BLOCK_R=self.BLOCK_R, BLOCK_F=block_f,
        )
        return out.view_as(x)
