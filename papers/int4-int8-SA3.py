import math
import gc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# W4A8 - Quantized Attention inspired from SageAttention3 (https://arxiv.org/pdf/2505.11594), since A10 doesn't have FP8 Tensorcores
# we go with INT8 cores

# imp constants 
INT4_MAX   = 7       
INT8_MAX   = 127.0  
GROUP_T    = 8       
LOG2_E     = 1.44269504
LOG2_127   = 6.98868468677


_int4_configs =[
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_stages=4, num_warps=4),
]


@triton.jit
def _smooth_quant_qk_kernel(
    Q_ptr, Qq_ptr, Qs_ptr, Qm_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_qqb, stride_qqh, stride_qqs, stride_qqd,
    stride_qsb, stride_qsh, stride_qsg, stride_qmb, stride_qmh,
    K_ptr, Kq_ptr, Ks_ptr, Km_ptr,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_kqb, stride_kqh, stride_kqs, stride_kqd,
    stride_ksb, stride_ksh, stride_ksg, stride_kmb, stride_kmh,
    B: tl.constexpr, 
    H: tl.constexpr, 
    S: tl.constexpr, 
    D: tl.constexpr,
    GROUP_T: tl.constexpr, 
    BLOCK_D: tl.constexpr, 
    BLOCK_S: tl.constexpr, 
    INT4_MAX: tl.constexpr,
):
    b, h = tl.program_id(0), tl.program_id(1)
    q_base  = b * stride_qb  + h * stride_qh
    k_base  = b * stride_kb  + h * stride_kh
    qq_base = b * stride_qqb + h * stride_qqh
    kq_base = b * stride_kqb + h * stride_kqh
    qs_base = b * stride_qsb + h * stride_qsh
    ks_base = b * stride_ksb + h * stride_ksh
    qm_base = b * stride_qmb + h * stride_qmh
    km_base = b * stride_kmb + h * stride_kmh

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    q_sum, k_sum = tl.zeros([BLOCK_D], dtype=tl.float32), tl.zeros([BLOCK_D], dtype=tl.float32)
    for s0 in range(0, S, BLOCK_S):
        s_offs = s0 + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        q_ptrs = Q_ptr + q_base + s_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
        k_ptrs = K_ptr + k_base + s_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        q_sum += tl.sum(tl.load(q_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0), axis=0)
        k_sum += tl.sum(tl.load(k_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0), axis=0)
        
    q_mean, k_mean = (q_sum / S).to(tl.float16), (k_sum / S).to(tl.float16)
    tl.store(Qm_ptr + qm_base + d_offs * stride_qd, q_mean, mask=d_mask)
    tl.store(Km_ptr + km_base + d_offs * stride_kd, k_mean, mask=d_mask)

    n_groups = tl.cdiv(S, GROUP_T)
    for g in range(0, n_groups):
        s_offs = g * GROUP_T + tl.arange(0, GROUP_T)
        s_mask = s_offs < S
        
        q_ptrs = Q_ptr  + q_base  + s_offs[:, None] * stride_qs  + d_offs[None, :] * stride_qd
        qq_ptrs = Qq_ptr + qq_base + s_offs[:, None] * stride_qqs + d_offs[None, :] * stride_qqd
        q_smoothed = tl.load(q_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0) - q_mean[None, :] 
        q_scale = tl.maximum(tl.max(tl.abs(q_smoothed)), 1e-8) / INT4_MAX
        q_q = tl.minimum(tl.maximum(tl.math.floor(q_smoothed / q_scale + 0.5), -7.0), 7.0).to(tl.int8)
        tl.store(qq_ptrs, q_q, mask=s_mask[:, None] & d_mask[None, :])
        tl.store(Qs_ptr + qs_base + g * stride_qsg, q_scale.to(tl.float32))

        k_ptrs = K_ptr  + k_base  + s_offs[:, None] * stride_ks  + d_offs[None, :] * stride_kd
        kq_ptrs = Kq_ptr + kq_base + s_offs[:, None] * stride_kqs + d_offs[None, :] * stride_kqd
        k_smoothed = tl.load(k_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0) - k_mean[None, :]
        k_scale = tl.maximum(tl.max(tl.abs(k_smoothed)), 1e-8) / INT4_MAX
        k_q = tl.minimum(tl.maximum(tl.math.floor(k_smoothed / k_scale + 0.5), -7.0), 7.0).to(tl.int8)
        tl.store(kq_ptrs, k_q, mask=s_mask[:, None] & d_mask[None, :])
        tl.store(Ks_ptr + ks_base + g * stride_ksg, k_scale.to(tl.float32))


def smooth_quantize_qk(q: torch.Tensor, k: torch.Tensor):
    B, H, S, D = q.shape
    G    = math.ceil(S / GROUP_T)
    BD   = triton.next_power_of_2(D)
    BS   = min(64, triton.next_power_of_2(S))

    q_int4, q_scale, q_mean = torch.empty_like(q, dtype=torch.int8), torch.empty((B, H, G), dtype=torch.float32, device=q.device), torch.empty((B, H, D), dtype=torch.float16, device=q.device)
    k_int4, k_scale, k_mean = torch.empty_like(k, dtype=torch.int8), torch.empty((B, H, G), dtype=torch.float32, device=k.device), torch.empty((B, H, D), dtype=torch.float16, device=k.device)

    _smooth_quant_qk_kernel[(B, H)](
        q, q_int4, q_scale, q_mean, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        q_int4.stride(0), q_int4.stride(1), q_int4.stride(2), q_int4.stride(3), q_scale.stride(0), q_scale.stride(1), q_scale.stride(2), q_mean.stride(0), q_mean.stride(1),
        k, k_int4, k_scale, k_mean, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        k_int4.stride(0), k_int4.stride(1), k_int4.stride(2), k_int4.stride(3), k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), k_mean.stride(0), k_mean.stride(1),
        B=B, H=H, S=S, D=D, GROUP_T=GROUP_T, BLOCK_D=BD, BLOCK_S=BS, INT4_MAX=INT4_MAX, num_warps=2,
    )
    return q_int4, q_scale, q_mean, k_int4, k_scale, k_mean



@triton.jit
def _quant_v_int8_kernel(
    V_ptr, Vq_ptr, Vs_ptr, Vm_ptr,
    stride_vb, stride_vh, stride_vs, stride_vd, 
    stride_vqb, stride_vqh, stride_vqs, stride_vqd,
    stride_vsb, stride_vsh, stride_vsd, stride_vmb, stride_vmh,
    S: tl.constexpr, D: tl.constexpr, 
    BLOCK_S: tl.constexpr, smooth_v: tl.constexpr, 
    INT8_MAX: tl.constexpr,
):
    b, h, d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    v_base, vq_base = b * stride_vb + h * stride_vh, b * stride_vqb + h * stride_vqh
    v_mean = tl.zeros([1], dtype=tl.float32)
    if smooth_v:
        for s0 in range(0, S, BLOCK_S):
            s_offs = s0 + tl.arange(0, BLOCK_S)
            vals = tl.load(V_ptr + v_base + s_offs * stride_vs + d * stride_vd, mask=s_offs < S, other=0.0)
            v_mean += tl.sum(vals.to(tl.float32), axis=0)
        v_mean = v_mean / S
        tl.store(Vm_ptr + b * stride_vmb + h * stride_vmh + d + tl.arange(0, 1), v_mean.to(tl.float16))

    amax = tl.zeros([1], dtype=tl.float32) + 1e-8
    for s0 in range(0, S, BLOCK_S):
        s_offs = s0 + tl.arange(0, BLOCK_S)
        vals = tl.load(V_ptr + v_base + s_offs * stride_vs + d * stride_vd, mask=s_offs < S, other=0.0).to(tl.float32)
        if smooth_v: vals = vals - v_mean
        amax = tl.maximum(amax, tl.max(tl.abs(vals)))

    scale = amax / INT8_MAX
    tl.store(Vs_ptr + b * stride_vsb + h * stride_vsh + d * stride_vsd + tl.arange(0, 1), scale.to(tl.float32))

    for s0 in range(0, S, BLOCK_S):
        s_offs = s0 + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        vals = tl.load(V_ptr + v_base + s_offs * stride_vs + d * stride_vd, mask=s_mask, other=0.0).to(tl.float32)
        if smooth_v: vals = vals - v_mean
        v_q = tl.minimum(tl.maximum(tl.math.floor(vals / scale + 0.5), -INT8_MAX), INT8_MAX).to(tl.int8)
        tl.store(Vq_ptr + vq_base + s_offs * stride_vqs + d * stride_vqd, v_q, mask=s_mask)



def quantize_v_int8(v: torch.Tensor, smooth_v: bool = False):
    B, H, S, D = v.shape
    v_int8 = torch.empty_like(v, dtype=torch.int8)
    v_scale = torch.empty((B, H, D), dtype=torch.float32, device=v.device)
    v_mean_t = torch.empty((B, H, D), dtype=torch.float16, device=v.device) if smooth_v else None
    vm_ptr = v_mean_t if smooth_v else v

    _quant_v_int8_kernel[(B, H, D)](
        v, v_int8, v_scale, vm_ptr, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        (vm_ptr.stride(0) if smooth_v else 0), (vm_ptr.stride(1) if smooth_v else 0),
        S=S, D=D, BLOCK_S=min(64, triton.next_power_of_2(S)), smooth_v=smooth_v, INT8_MAX=INT8_MAX, num_warps=1,
    )
    return v_int8, v_scale, v_mean_t


@triton.jit
def _attn_inner_int4_int8(
    acc, l_i, m_i,
    q_int4, q_s_expanded,
    K_block_ptr, K_scale_ptr_base,
    V_block_ptr, DS_ptr_base,
    start_m,
    BLOCK_M: tl.constexpr, 
    HEAD_DIM: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    GROUP_T: tl.constexpr,
    STAGE: tl.constexpr, 
    offs_m: tl.constexpr, offs_n: tl.constexpr, 
    N_CTX: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, N_CTX
    elif STAGE == 2:
        lo, hi = tl.multiple_of(start_m * BLOCK_M, BLOCK_M), (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, start_m * BLOCK_M

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    k_grp = tl.arange(0, BLOCK_N) // GROUP_T

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_int4 = tl.load(K_block_ptr)

        # QK = INT4 X INT4 -> INT32 
        qk_i32 = tl.dot(q_int4, k_int4, out_dtype=tl.int32)
        
        # Zero-cost native 1D Load & Broadcast 
        k_s = tl.load(K_scale_ptr_base + (start_n // GROUP_T) + k_grp)[None, :]
        ds  = tl.load(DS_ptr_base + start_n + offs_n)[None, :]

        # QK = INT32 -> FP32 
        qk = qk_i32.to(tl.float32) * q_s_expanded * k_s + ds

        if STAGE == 2:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        
        # we avoid 127.0 divison overhead, and skip FP32 * 127.0 
        p = tl.math.exp2(qk - m_ij[:, None] + 6.98868468677)  # log2(127)
        l_ij = tl.sum(p, 1)
        p_int8 = p.to(tl.int8) # map softmax prob values to INT8 

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_int8 = tl.load(V_block_ptr)
        acc += tl.dot(p_int8, v_int8, out_dtype=tl.int32).to(tl.float32) 

        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


@triton.autotune(_int4_configs, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd_int4_int8(
    Q, K, Q_scale, K_scale, Delta_S, V, V_scale, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_on,
    stride_qsb, stride_qsh, stride_qsg,
    stride_ksb, stride_ksh, stride_ksg,
    stride_vsb, stride_vsh, stride_vsd,
    stride_dsb, stride_dsh, stride_dsn,
    Z, H, N_CTX, qk_scale,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_T: tl.constexpr, STAGE: tl.constexpr,
):
    start_m, off_hz = tl.program_id(0), tl.program_id(1)
    off_z, off_h = off_hz // H, off_hz % H

    qk_offset = off_z.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    qs_offset = off_z.to(tl.int64) * stride_qsb + off_h.to(tl.int64) * stride_qsh
    ks_offset = off_z.to(tl.int64) * stride_ksb + off_h.to(tl.int64) * stride_ksh
    vs_offset = off_z.to(tl.int64) * stride_vsb + off_h.to(tl.int64) * stride_vsh
    ds_offset = off_z.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(
        base=K + qk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1)) # Col-Major SMEM!
    V_block_ptr = tl.make_block_ptr(
        base=V + qk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vn, stride_vk),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(0, 1)) # Col-Major SMEM!

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 1D Scale Loading Outside Loop
    q_s_ptrs = Q_scale + qs_offset + (start_m * BLOCK_M // GROUP_T) + (tl.arange(0, BLOCK_M) // GROUP_T)
    q_s_expanded = (tl.load(q_s_ptrs) * qk_scale)[:, None]

    K_scale_ptr_base = K_scale + ks_offset
    DS_ptr_base = Delta_S + ds_offset

    q_int4 = tl.load(Q_block_ptr)       

    if STAGE & 1:
        acc, l_i, m_i = _attn_inner_int4_int8(
            acc, l_i, m_i, q_int4, q_s_expanded, K_block_ptr, K_scale_ptr_base, V_block_ptr, DS_ptr_base,
            start_m, BLOCK_M, HEAD_DIM, BLOCK_N, GROUP_T, 1, offs_m, offs_n, N_CTX)
    if STAGE & 2:
        acc, l_i, m_i = _attn_inner_int4_int8(
            acc, l_i, m_i, q_int4, q_s_expanded, K_block_ptr, K_scale_ptr_base, V_block_ptr, DS_ptr_base,
            start_m, BLOCK_M, HEAD_DIM, BLOCK_N, GROUP_T, 3, offs_m, offs_n, N_CTX)
        acc, l_i, m_i = _attn_inner_int4_int8(
            acc, l_i, m_i, q_int4, q_s_expanded, K_block_ptr, K_scale_ptr_base, V_block_ptr, DS_ptr_base,
            start_m, BLOCK_M, HEAD_DIM, BLOCK_N, GROUP_T, 2, offs_m, offs_n, N_CTX)

    acc = acc / l_i[:, None]
    
    v_scale_val = tl.load(V_scale + vs_offset + tl.arange(0, HEAD_DIM))
    acc = acc * v_scale_val[None, :]

    O_block_ptr = tl.make_block_ptr(
        base=Out + qk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))



def int4_flash_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    sm_scale: float, causal: bool = False, 
    smooth_v: bool = False
) -> torch.Tensor:

    B, H, S, D = q.shape

    q_int4, q_scale, q_mean, k_int4, k_scale, k_mean = smooth_quantize_qk(q.float(), k.float())
    
    qk_scale = sm_scale * LOG2_E
    k_smoothed = k.half() - k_mean.unsqueeze(2)           
    delta_s = torch.matmul(q_mean.unsqueeze(2).to(torch.float16), k_smoothed.transpose(-1, -2)).squeeze(2).contiguous() * qk_scale
    
    v_int8, v_scale, v_mean = quantize_v_int8(v.float(), smooth_v=smooth_v)
    v_int8 = v_int8.transpose(2, 3).contiguous() 

    o = torch.empty((B, H, S, D), dtype=torch.float16, device=q.device)

    stage = 3 if causal else 1
    grid  = lambda args: (triton.cdiv(S, args["BLOCK_M"]), B * H, 1)

    _attn_fwd_int4_int8[grid](
        q_int4, k_int4, q_scale, k_scale, delta_s, v_int8, v_scale, o,
        q_int4.stride(0), q_int4.stride(1), q_int4.stride(2), q_int4.stride(3),
        k_int4.stride(0), k_int4.stride(1), k_int4.stride(2), k_int4.stride(3),
        v_int8.stride(0), v_int8.stride(1), v_int8.stride(3), v_int8.stride(2), 
        o.stride(0),      o.stride(1),      o.stride(2),      o.stride(3),
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        delta_s.stride(0), delta_s.stride(1), delta_s.stride(2),
        Z=B, H=H, N_CTX=S, qk_scale=qk_scale, HEAD_DIM=D, GROUP_T=GROUP_T, STAGE=stage,
    )

    if smooth_v and v_mean is not None:
        o = o + v_mean.unsqueeze(2).half()

    return o



# Bench 

def _hbm_bytes_int4(B, H, S, D, BLOCK_M=128):
    inner_loops = S / BLOCK_M
    return float(
        B * H * S * D * 1 + B * H * (S // GROUP_T) * 4 +  
        (B * H * S * D * 1 + B * H * (S // GROUP_T) * 4 + B * H * S * 2) * inner_loops +
        B * H * S * D * 1 * inner_loops + B * H * S * D * 2 
    )


def benchmark_kernel(fn, warmup=25, repeats=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeats


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)).item()


def main():
    device = torch.device("cuda")
    print(f"\n{'='*88}\n  INT4+INT8 Flash Attention \n{'='*88}")

    bench_configs =[
        dict(batch=1, seq_len=512,   embed=1024, heads=16),
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

    hdr = (f"{'Config':<22} | {'Impl':<16} | {'ms':>7} | "
           f"{'TFLOPS':>8} | {'BW GB/s':>9} | {'Cosine':>8} | {'Speedup':>8}")
    print(hdr)
    print("-" * len(hdr))

    for cfg in bench_configs:
        B, S, E, H_heads = cfg["batch"], cfg["seq_len"], cfg["embed"], cfg["heads"]
        D = E // H_heads
        sm_scale = 1.0 / math.sqrt(D)
        
        q = torch.randn(B, H_heads, S, D, device=device), 
        k = torch.randn(B, H_heads, S, D, device=device), 
        v = torch.randn(B, H_heads, S, D, device=device)
        ref = F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=False)

        # FP16 Benchmark
        ms_fp16 = benchmark_kernel(lambda: F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=False))
        
        with torch.no_grad(): out_i4 = int4_flash_attention(q, k, v, sm_scale, causal=False)
        cos = cosine_sim(ref, out_i4)

        q_int4, q_scale, q_mean, k_int4, k_scale, k_mean = smooth_quantize_qk(q, k)
        k_smoothed = k.half() - k_mean.unsqueeze(2)
        qk_scale = sm_scale * LOG2_E
        delta_s_scaled = torch.matmul(q_mean.unsqueeze(2).to(torch.float16), k_smoothed.transpose(-1, -2)).squeeze(2).contiguous() * qk_scale
        
        v_int8, v_scale, _ = quantize_v_int8(v, smooth_v=False)
        v_int8_t = v_int8.transpose(2, 3).contiguous()  # Transpose V
        
        O_bench = torch.empty((B, H_heads, S, D), dtype=torch.float16, device=device)
        grid = lambda args: (triton.cdiv(S, args["BLOCK_M"]), B * H_heads, 1)

        def kernel_only():
            _attn_fwd_int4_int8[grid](
                q_int4, k_int4, q_scale, k_scale, delta_s_scaled, v_int8_t, v_scale, O_bench,
                q_int4.stride(0), q_int4.stride(1), q_int4.stride(2), q_int4.stride(3),
                k_int4.stride(0), k_int4.stride(1), k_int4.stride(2), k_int4.stride(3),
                v_int8_t.stride(0), v_int8_t.stride(1), v_int8_t.stride(3), v_int8_t.stride(2),
                O_bench.stride(0), O_bench.stride(1), O_bench.stride(2), O_bench.stride(3),
                q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
                k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
                v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
                delta_s_scaled.stride(0), delta_s_scaled.stride(1), delta_s_scaled.stride(2),
                Z=B, H=H_heads, N_CTX=S, qk_scale=qk_scale, HEAD_DIM=D, GROUP_T=8, STAGE=1,
            )

        ms_i4 = benchmark_kernel(kernel_only) # INT4-INT8 kernel only bench 

        flops       = float(4 * B * H_heads * S * S * D) 
        tflops_fp16 = (flops / 1e12) / (ms_fp16 / 1000)
        tflops_i4   = (flops / 1e12) / (ms_i4  / 1000)
        bw_i4       = (_hbm_bytes_int4(B, H_heads, S, D) / 1e9) / (ms_i4 / 1000)

        tag = f"S={S} E={E}"
        print(f"{tag:<22} | {'FP16-SDPA':<16} | {ms_fp16:>7.3f} | {tflops_fp16:>8.2f} | {'—':>9} | {'—':>8} | {'1.00x':>8}")
        print(f"{'':<22} | {'INT4-Opt':<16} | {ms_i4:>7.3f} | {tflops_i4:>8.2f} | {bw_i4:>9.2f} | {cos:>8.6f} | {ms_fp16/ms_i4:>7.2f}x")

if __name__ == "__main__":
    main()



# Bench results 

# ========================================================================================
#   INT4+INT8 Flash Attention 
# ========================================================================================
# Config                 | Impl             |      ms |   TFLOPS |   BW GB/s |   Cosine |  Speedup
# ------------------------------------------------------------------------------------------------
# S=512 E=1024           | FP16-SDPA        |   0.054 |    19.72 |         — |        — |    1.00x
#                        | INT4-Opt         |   0.112 |     9.60 |     52.33 | 0.980322 |    0.49x

# S=1024 E=1024          | FP16-SDPA        |   0.140 |    30.58 |         — |        — |    1.00x
#                        | INT4-Opt         |   0.110 |    38.96 |    183.78 | 0.980531 |    1.27x

# S=2048 E=1024          | FP16-SDPA        |   0.433 |    39.69 |         — |        — |    1.00x
#                        | INT4-Opt         |   0.250 |    68.63 |    298.51 | 0.979533 |    1.73x

# S=4096 E=1024          | FP16-SDPA        |   1.355 |    50.71 |         — |        — |    1.00x
#                        | INT4-Opt         |   0.838 |    82.00 |    341.62 | 0.978944 |    1.62x

# S=8192 E=1024          | FP16-SDPA        |   4.704 |    58.44 |         — |        — |    1.00x
#                        | INT4-Opt         |   3.022 |    90.94 |    370.54 | 0.977712 |    1.56x

# S=16384 E=1024         | FP16-SDPA        |  18.090 |    60.78 |         — |        — |    1.00x
#                        | INT4-Opt         |  11.914 |    92.29 |    371.79 | 0.976433 |    1.52x

# S=1024 E=2048          | FP16-SDPA        |   0.270 |    31.85 |         — |        — |    1.00x
#                        | INT4-Opt         |   0.166 |    51.68 |    241.74 | 0.977673 |    1.62x

# S=2048 E=2048          | FP16-SDPA        |   0.857 |    40.11 |         — |        — |    1.00x
#                        | INT4-Opt         |   0.614 |    55.97 |    241.29 | 0.977662 |    1.40x

# S=4096 E=2048          | FP16-SDPA        |   2.691 |    51.07 |         — |        — |    1.00x
#                        | INT4-Opt         |   2.077 |    66.17 |    273.15 | 0.976825 |    1.30x

# S=8192 E=2048          | FP16-SDPA        |   9.360 |    58.73 |         — |        — |    1.00x
#                        | INT4-Opt         |   7.606 |    72.28 |    291.74 | 0.975140 |    1.23x

# S=16384 E=2048         | FP16-SDPA        |  35.999 |    61.08 |         — |        — |    1.00x
#                        | INT4-Opt         |  30.213 |    72.78 |    290.42 | 0.974552 |    1.19x

# S=32768 E=2048         | FP16-SDPA        | 141.130 |    62.33 |         — |        — |    1.00x
#                        | INT4-Opt         | 120.524 |    72.98 |    289.54 | 0.973376 |    1.17x