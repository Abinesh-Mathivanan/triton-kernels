import torch
import triton
import triton.language as tl

@triton.jit
def tc_xxt_kernel(
    A_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_idx = k + rk

        a = tl.load(
            A_ptr + rm[:, None] * stride_am + k_idx[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (k_idx[None, :] < K),
            other=0.0
        )

        b = tl.load(
            A_ptr + k_idx[:, None] * stride_ak + rn[None, :] * stride_am,
            mask=(k_idx[:, None] < K) & (rn[None, :] < M),
            other=0.0
        )

        acc += tl.dot(
            a.to(tl.float16),  
            b.to(tl.float16)
        )

    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < M)
    )


def xxt_tensorcore(X: torch.Tensor):
    assert X.is_cuda

    M, K = X.shape

    X_tc = X.to(torch.float16)

    C = torch.zeros((M, M), device=X.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(M, BLOCK_N),
    )

    tc_xxt_kernel[grid](
        X_tc, C,
        M, K,
        X_tc.stride(0), X_tc.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C


def baseline(X):
    return X @ X.T


def check_correctness():
    print("=" * 60)
    print("CORRECTNESS CHECK (Tensor Core)")
    print("=" * 60)

    torch.manual_seed(0)

    X = torch.randn((512, 512), device="cuda", dtype=torch.float32)

    ref = baseline(X)
    out = xxt_tensorcore(X)

    diff = (ref - out)

    max_abs = diff.abs().max().item()
    rel = torch.norm(diff) / torch.norm(ref)

    print(f"max_abs  = {max_abs:.2e}")
    print(f"frob_rel = {rel:.2e}")

    if rel < 1e-2:
        print("Correct (TC tolerance)")
    else:
        print("Mismatch")


def benchmark():
    print("\n" + "=" * 60)
    print("BENCHMARK (8192x8192)")
    print("=" * 60)

    X = torch.randn((8192, 8192), device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(5):
        baseline(X)
        xxt_tensorcore(X)

    torch.cuda.synchronize()

    # Torch (cuBLAS)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(50):
        baseline(X)
    end.record()
    torch.cuda.synchronize()
    t_torch = start.elapsed_time(end) / 50

    # Tensor Core
    start.record()
    for _ in range(50):
        xxt_tensorcore(X)
    end.record()
    torch.cuda.synchronize()
    t_triton = start.elapsed_time(end) / 50

    print(f"cuBLAS (torch) : {t_torch:.3f} ms")
    print(f"Triton TC      : {t_triton:.3f} ms")


if __name__ == "__main__":
    check_correctness()
    benchmark()
