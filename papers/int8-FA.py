import math
import gc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# paper reference - https://arxiv.org/abs/2409.16997 (INT-FlashAttention: Enabling Flash Attention for INT8 Quantization)

def keep(conf):
    return conf.kwargs["BLOCK_N"] <= conf.kwargs.get("HEAD_DIM", 64)

configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_stages=5, num_warps=4),
]


# Per-token quantization (Q and K): scale = amax / 127, stored per (B, H, S)
@triton.jit
def _quant_per_token_kernel(
    x_ptr, q_ptr, s_ptr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_sb, stride_sh, stride_ss,
    D: tl.constexpr, BLOCK_D: tl.constexpr
):
    b, h, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    x_off = b * stride_xb + h * stride_xh + s * stride_xs
    q_off = b * stride_qb + h * stride_qh + s * stride_qs
    s_off = b * stride_sb + h * stride_sh + s * stride_ss

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(x_ptr + x_off + cols * stride_xd, mask=mask, other=0.0)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-8)
    scale = amax / 127.0

    scaled = x / scale
    x_q = tl.math.floor(tl.abs(scaled) + 0.5) * tl.where(scaled >= 0, 1.0, -1.0)
    x_q = tl.minimum(tl.maximum(x_q, -127.0), 127.0).to(tl.int8)

    tl.store(q_ptr + q_off + cols * stride_qd, x_q, mask=mask)
    tl.store(s_ptr + s_off, scale)



def quantize_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = x.shape
    out_q = torch.empty((B, H, S, D), dtype=torch.int8, device=x.device)
    scale = torch.empty((B, H, S), dtype=torch.float32, device=x.device)
    BD = triton.next_power_of_2(D)

    _quant_per_token_kernel[(B, H, S)](
        x, out_q, scale,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2),
        D, BLOCK_D=BD, num_warps=1 if BD <= 64 else 2
    )
    return out_q, scale


# Per-head quantization (V): single scale per (B, H) covering all tokens.
@triton.jit
def _quant_per_head_kernel(
    v_ptr, vq_ptr, vs_ptr,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_vqb, stride_vqh, stride_vqs, stride_vqd,
    stride_vsb, stride_vsh,
    S: tl.constexpr, D: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    b, h = tl.program_id(0), tl.program_id(1)

    # Pass 1: find amax across all (s, d)
    amax = 0.0
    for s_idx in range(0, S, BLOCK_S):
        offs_s = s_idx + tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_s[:, None] < S) & (offs_d[None, :] < D)
        v_ptrs = v_ptr + b * stride_vb + h * stride_vh \
                 + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask, other=0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(v)))

    amax  = tl.maximum(amax, 1e-8)
    scale = amax / 127.0
    tl.store(vs_ptr + b * stride_vsb + h * stride_vsh, scale)


    # Pass 2: quantize 
    for s_idx in range(0, S, BLOCK_S):
        offs_s = s_idx + tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_s[:, None] < S) & (offs_d[None, :] < D)
        v_ptrs  = v_ptr  + b * stride_vb  + h * stride_vh  \
                  + offs_s[:, None] * stride_vs  + offs_d[None, :] * stride_vd
        vq_ptrs = vq_ptr + b * stride_vqb + h * stride_vqh \
                  + offs_s[:, None] * stride_vqs + offs_d[None, :] * stride_vqd
        v       = tl.load(v_ptrs, mask=mask, other=0.0)
        scaled  = v / scale
        v_q     = tl.math.floor(tl.abs(scaled) + 0.5) * tl.where(scaled >= 0, 1.0, -1.0)
        v_q     = tl.minimum(tl.maximum(v_q, -127.0), 127.0).to(tl.int8)
        tl.store(vq_ptrs, v_q, mask=mask)


def quantize_per_head(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = v.shape
    out_q = torch.empty((B, H, S, D), dtype=torch.int8, device=v.device)
    scale = torch.empty((B, H), dtype=torch.float32, device=v.device)
    BD = triton.next_power_of_2(D)
    # BLOCK_S must evenly cover S in the worst case; 64 is a safe default.
    BLOCK_S = min(64, triton.next_power_of_2(S))

    _quant_per_head_kernel[(B, H)](
        v, out_q, scale,
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
        scale.stride(0), scale.stride(1),
        S, D, BLOCK_S=BLOCK_S, BLOCK_D=BD, num_warps=2
    )
    return out_q, scale


@triton.jit
def _attn_fwd_inner_int8(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    q_scale, K_block_scale_ptr, v_scale,
    start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    N_CTX: tl.constexpr
):

    # STAGE=1: non-causal full attention  (lo=0,        hi=N_CTX)
    # STAGE=2: causal diagonal block      (lo=start_m,  hi=start_m+1) * BLOCK
    # STAGE=3: causal prefix blocks       (lo=0,        hi=start_m)   * BLOCK
    # (Calling convention: the outer kernel calls STAGE=3 first then STAGE=2, matching the original split-K causal pattern.)


    if STAGE == 1:
        lo, hi = 0, N_CTX
    elif STAGE == 2:
        # Diagonal block: tokens [start_m*BM, (start_m+1)*BM) can see only tokens up to and including their own position.
        lo = start_m * BLOCK_M
        hi = (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:  
        lo, hi = 0, start_m * BLOCK_M


    K_block_ptr      = tl.advance(K_block_ptr,       (0, lo))
    K_block_scale_ptr = tl.advance(K_block_scale_ptr, (lo,))
    V_block_ptr      = tl.advance(V_block_ptr,        (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # QK matmul (INT8 x INT8 -> INT32) 
        k       = tl.load(K_block_ptr)
        k_scale = tl.load(K_block_scale_ptr)

        qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
        # Dequantize: each element qk[i,j] = (q_int8[i] * q_s[i]) · (k_int8[j] * k_s[j])
        qk = qk * q_scale[:, None] * k_scale[None, :]

        qk = qk * qk_scale  # apply sm_scale * log2(e) for exp2


        if STAGE == 2:
            # Only tokens with col <= row are valid (causal)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk   = tl.where(mask, qk, -1.0e6)

        # we perform online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk   = qk - m_ij[:, None]
        p    = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # Cast p to INT8 for the PV matmul 
        # p ∈ [0, 1] after the softmax shift, so p * 127 ∈ [0, 127].
        # floor(x + 0.5) gives nearest-integer rounding for x >= 0.
        # i tried to avoid division as much as possible
        p_int8 = tl.math.floor(p * 127.0 + 0.5).to(tl.int8)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i   = l_i * alpha + l_ij
        acc   = acc * alpha[:, None]


        # PV matmul (INT8 x INT8 -> INT32)
        v   = tl.load(V_block_ptr)
        pv  = tl.dot(p_int8, v, out_dtype=tl.int32).to(tl.float32)
        # Dequantize: the p_int8 scale is (1 / 127), v_scale is the per-head scale.
        acc += pv * (v_scale / 127.0)

        m_i = m_ij
        V_block_ptr       = tl.advance(V_block_ptr,        (BLOCK_N, 0))
        K_block_ptr       = tl.advance(K_block_ptr,        (0, BLOCK_N))
        K_block_scale_ptr = tl.advance(K_block_scale_ptr,  (BLOCK_N,))

    return acc, l_i, m_i


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd_int8(
    Q, K, V, Q_scale, K_scale, V_scale, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_sqz, stride_sqh, stride_sqm,
    stride_vsz, stride_vsh,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m, off_hz = tl.program_id(0), tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_offset  = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    scl_offset  = off_z.to(tl.int64) * stride_sqz + off_h.to(tl.int64) * stride_sqh
    vscl_offset = off_z.to(tl.int64) * stride_vsz + off_h.to(tl.int64) * stride_vsh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))


    # Q scale: per-token, shape (B, H, S)
    Q_block_scale_ptr = tl.make_block_ptr(
        base=Q_scale + scl_offset, shape=(N_CTX,),
        strides=(stride_sqm,), offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,), order=(0,))

    # K scale: same layout as Q scale (per-token, same strides)
    K_block_scale_ptr = tl.make_block_ptr(
        base=K_scale + scl_offset, shape=(N_CTX,),
        strides=(stride_sqm,), offsets=(0,),
        block_shape=(BLOCK_N,), order=(0,))

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)


    # Pre-multiply sm_scale by log2(e) so we can use exp2 instead of exp
    qk_scale = sm_scale * 1.44269504

    q       = tl.load(Q_block_ptr)
    q_scale = tl.load(Q_block_scale_ptr)
    v_scale = tl.load(V_scale + vscl_offset)

    # STAGE=1: full (non-causal) attention
    # STAGE=3: causal — run prefix blocks (STAGE=3 inner) then diagonal (STAGE=2 inner)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_int8(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            q_scale, K_block_scale_ptr, v_scale,
            start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
            1, offs_m, offs_n, N_CTX)

    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner_int8(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            q_scale, K_block_scale_ptr, v_scale,
            start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
            3, offs_m, offs_n, N_CTX)
        acc, l_i, m_i = _attn_fwd_inner_int8(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            q_scale, K_block_scale_ptr, v_scale,
            start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
            2, offs_m, offs_n, N_CTX)

    m_i += tl.math.log2(l_i)
    acc   = acc / l_i[:, None]

    tl.store(M + off_hz * N_CTX + offs_m, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def int8_flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    sm_scale: float, causal: bool = False
) -> torch.Tensor:
    B, H, S, D = q.shape
    q_q, q_s = quantize_per_token(q)
    k_q, k_s = quantize_per_token(k)
    v_q, v_s = quantize_per_head(v)

    o = torch.empty((B, H, S, D), dtype=torch.float16, device=q.device)
    M = torch.empty((B * H, S), device=q.device, dtype=torch.float32)

    # STAGE encoding: 1 = full attention, 3 = causal (prefix + diagonal)
    stage = 3 if causal else 1

    grid = lambda args: (triton.cdiv(S, args["BLOCK_M"]), B * H, 1)

    _attn_fwd_int8[grid](
        q_q, k_q, v_q, q_s, k_s, v_s, sm_scale, M, o,
        q_q.stride(0), q_q.stride(1), q_q.stride(2), q_q.stride(3),
        k_q.stride(0), k_q.stride(1), k_q.stride(2), k_q.stride(3),
        v_q.stride(0), v_q.stride(1), v_q.stride(2), v_q.stride(3),
        o.stride(0),   o.stride(1),   o.stride(2),   o.stride(3),
        q_s.stride(0), q_s.stride(1), q_s.stride(2),
        v_s.stride(0), v_s.stride(1),
        B, H, S, HEAD_DIM=D, STAGE=stage,
    )
    return o


# FP16 - FLASH SDPA
class FP16FlashAttentionSDPA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, causal: bool = False):
        super().__init__()
        self.H, self.D, self.ca = num_heads, embed_dim // num_heads, causal
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        xf = x.float()
        q = self.q_proj(xf).reshape(B, S, self.H, self.D).transpose(1, 2).half()
        k = self.k_proj(xf).reshape(B, S, self.H, self.D).transpose(1, 2).half()
        v = self.v_proj(xf).reshape(B, S, self.H, self.D).transpose(1, 2).half()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.ca)
        return self.o_proj(out.float().transpose(1, 2).reshape(B, S, C)).to(x.dtype)


class INTFlashAttentionTriton(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, causal: bool = False):
        super().__init__()
        self.H, self.D, self.ca = num_heads, embed_dim // num_heads, causal
        self.sm_scale = 1.0 / math.sqrt(self.D)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        xf = x.float()
        q = self.q_proj(xf).reshape(B, S, self.H, self.D).transpose(1, 2)
        k = self.k_proj(xf).reshape(B, S, self.H, self.D).transpose(1, 2)
        v = self.v_proj(xf).reshape(B, S, self.H, self.D).transpose(1, 2)
        out = int8_flash_attention(q, k, v, self.sm_scale, self.ca)
        return self.o_proj(out.float().transpose(1, 2).reshape(B, S, C)).to(x.dtype)


@dataclass
class BenchResult:
    name: str; seq_len: int; batch: int; heads: int; head_dim: int
    avg_ms: float; memory_mb: float; flops: float; tflops: float; bandwidth_gbs: float
    quant_error_cosine: Optional[float] = None


A10G_FP16_TFLOPS, A10G_INT8_TOPS = 125.0, 250.0


def _attn_flops(B, H, S, D):
    # QK: B*H * S * S * 2D  (matmul)  +  PV: B*H * S * S * 2D  (matmul)
    return float(2 * B * H * S * S * 2 * D)


def _hbm_bytes(B, H, S, D, bpe):
    # Q, K, V reads + O write, each (B, H, S, D) elements
    return float(4 * B * H * S * D * bpe)


def _hbm_bytes_int8(B, H, S, D):
    elem   = B * H * S * D
    scale  = B * H * S
    o_elem = elem
    return float(3 * elem * 1 + 2 * scale * 4 + o_elem * 2 + scale * 4)


def run_benchmark(
    model, x, B, S, E, label="", warmup=20, repeats=100,
    ref_out=None, bytes_per_elem=2.0
) -> BenchResult:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    mb_before = torch.cuda.max_memory_allocated() / 1024**2

    with torch.no_grad():
        out = model(x)
        mb = (torch.cuda.max_memory_allocated() / 1024**2) - mb_before
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); model(x); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

    avg = sum(times) / len(times)
    H, D = model.H, model.D
    fl = _attn_flops(B, H, S, D)

    cos = None
    if ref_out is not None:
        r = ref_out.float().flatten()
        a = out.float().flatten()
        cos = F.cosine_similarity(r.unsqueeze(0), a.unsqueeze(0)).item()

    return BenchResult(
        name=label, seq_len=S, batch=B, heads=H, head_dim=D,
        avg_ms=avg, memory_mb=mb, flops=fl,
        tflops=(fl / 1e12) / (avg / 1000),
        bandwidth_gbs=(_hbm_bytes(B, H, S, D, bytes_per_elem) / 1e9) / (avg / 1000),
        quant_error_cosine=cos,
    )


def _measure_cosine(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    r = ref.float().flatten()
    a = hyp.float().flatten()
    return F.cosine_similarity(r.unsqueeze(0), a.unsqueeze(0)).item()


def main():
    device = torch.device("cuda")
    print(f"\n{'='*80}")
    print(" Triton INT8 Flash Attention  (kernel-only, pre-quantized)")
    print(f"{'='*80}")

    bench_configs = [
        dict(batch=1, seq_len=1024,  embed=1024, heads=16),
        dict(batch=1, seq_len=2048,  embed=1024, heads=16),
        dict(batch=1, seq_len=4096,  embed=1024, heads=16),
        dict(batch=1, seq_len=8192,  embed=1024, heads=16),
        dict(batch=1, seq_len=16384, embed=1024, heads=16),
        dict(batch=1, seq_len=1024,  embed=2048, heads=16),
        dict(batch=1, seq_len=2048,  embed=2048, heads=16),
        dict(batch=1, seq_len=4096,  embed=2048, heads=16),
        dict(batch=1, seq_len=8192,  embed=2048, heads=16),
        dict(batch=1, seq_len=16384, embed=2048, heads=16),
        dict(batch=1, seq_len=32768, embed=2048, heads=16),
    ]

    hdr = f"{'Config':<22} | {'Impl':<14} | {'ms':>8} | {'TFLOPS':>8} | {'BW GB/s':>9} | {'Cosine':>8} | {'Speedup':>8}"
    print(hdr)
    print("-" * len(hdr))

    for cfg in bench_configs:
        B, S, E, H = cfg["batch"], cfg["seq_len"], cfg["embed"], cfg["heads"]
        D = E // H
        sm_scale = 1.0 / math.sqrt(D)

        # Inputs (raw FP32)
        q = torch.randn(B, H, S, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        v = torch.randn(B, H, S, D, device=device)

        # pre-quantize once
        with torch.no_grad():
            q_q, q_s = quantize_per_token(q)
            k_q, k_s = quantize_per_token(k)
            v_q, v_s = quantize_per_head(v)

        # reference output from SDPA to compute cosine sim 
        with torch.no_grad():
            ref_out = F.scaled_dot_product_attention(
                q.half(), k.half(), v.half(), is_causal=False)

        for _ in range(20):
            int8_flash_attention(q_q, k_q, v_q, sm_scale, causal=False)
            F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=False)
        torch.cuda.synchronize()


        # bench - INT8
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(100):
            int8_flash_attention(q_q, k_q, v_q, sm_scale, causal=False)
        e.record(); torch.cuda.synchronize()
        ms_i8 = s.elapsed_time(e) / 100

        # bench - FP16 SDPA
        s.record()
        for _ in range(100):
            F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=False)
        e.record(); torch.cuda.synchronize()
        ms_fp16 = s.elapsed_time(e) / 100

        with torch.no_grad():
            int8_out = int8_flash_attention(q_q, k_q, v_q, sm_scale, causal=False)
        cos = _measure_cosine(ref_out, int8_out)


        flops      = _attn_flops(B, H, S, D)
        tflops_i8  = (flops / 1e12) / (ms_i8  / 1000)
        tflops_fp16 = (flops / 1e12) / (ms_fp16 / 1000)
        bw_i8      = (_hbm_bytes_int8(B, H, S, D) / 1e9) / (ms_i8 / 1000)
        speedup    = ms_fp16 / ms_i8

        tag = f"S={S} E={E}"
        print(f"{tag:<22} | {'FP16-SDPA':<14} | {ms_fp16:>8.3f} | {tflops_fp16:>8.2f} | {'—':>9} | {'—':>8} | {'1.00x':>8}")
        print(f"{'':<22} | {'INT8-Triton':<14} | {ms_i8:>8.3f} | {tflops_i8:>8.2f} | {bw_i8:>9.2f} | {cos:>8.6f} | {speedup:>7.2f}x")
        print("-" * len(hdr))



if __name__ == "__main__":
    main()


# Bench results 

# ================================================================================
#  Triton INT8 Flash Attention  (kernel-only, pre-quantized)
# ================================================================================
# Config                 | Impl           |       ms |   TFLOPS |   BW GB/s |   Cosine |  Speedup
# -----------------------------------------------------------------------------------------------
# S=1024 E=1024          | FP16-SDPA      |    0.141 |    30.52 |         — |        — |    1.00x
#                        | INT8-Triton    |    0.291 |    14.76 |     18.69 | 0.252912 |    0.48x
# -----------------------------------------------------------------------------------------------
# S=2048 E=1024          | FP16-SDPA      |    0.433 |    39.65 |         — |        — |    1.00x
#                        | INT8-Triton    |    0.420 |    40.92 |     25.91 | 0.207001 |    1.03x
# -----------------------------------------------------------------------------------------------
# S=4096 E=1024          | FP16-SDPA      |    1.355 |    50.72 |         — |        — |    1.00x
#                        | INT8-Triton    |    1.281 |    53.63 |     16.98 | 0.166719 |    1.06x
# -----------------------------------------------------------------------------------------------
# S=8192 E=1024          | FP16-SDPA      |    4.705 |    58.42 |         — |        — |    1.00x
#                        | INT8-Triton    |    4.225 |    65.05 |     10.30 | 0.135431 |    1.11x
# -----------------------------------------------------------------------------------------------
# S=16384 E=1024         | FP16-SDPA      |   18.091 |    60.78 |         — |        — |    1.00x
#                        | INT8-Triton    |   15.635 |    70.33 |      5.57 | 0.109106 |    1.16x
# -----------------------------------------------------------------------------------------------
# S=1024 E=2048          | FP16-SDPA      |    0.270 |    31.82 |         — |        — |    1.00x
#                        | INT8-Triton    |    0.294 |    29.27 |     36.39 | 0.270287 |    0.92x
# -----------------------------------------------------------------------------------------------
# S=2048 E=2048          | FP16-SDPA      |    0.857 |    40.10 |         — |        — |    1.00x
#                        | INT8-Triton    |    0.679 |    50.58 |     31.45 | 0.224312 |    1.26x
# -----------------------------------------------------------------------------------------------
# S=4096 E=2048          | FP16-SDPA      |    2.692 |    51.06 |         — |        — |    1.00x
#                        | INT8-Triton    |    2.020 |    68.05 |     21.16 | 0.184351 |    1.33x
# -----------------------------------------------------------------------------------------------
# S=8192 E=2048          | FP16-SDPA      |    9.362 |    58.72 |         — |        — |    1.00x
#                        | INT8-Triton    |    6.657 |    82.58 |     12.84 | 0.150149 |    1.41x
# -----------------------------------------------------------------------------------------------
# S=16384 E=2048         | FP16-SDPA      |   36.000 |    61.08 |         — |        — |    1.00x
#                        | INT8-Triton    |   24.739 |    88.89 |      6.91 | 0.122045 |    1.46x
# -----------------------------------------------------------------------------------------------
# S=32768 E=2048         | FP16-SDPA      |  141.118 |    62.33 |         — |        — |    1.00x
#                        | INT8-Triton    |   95.230 |    92.37 |      3.59 | 0.098287 |    1.48x
# -----------------------------------------------------------------------------------------------