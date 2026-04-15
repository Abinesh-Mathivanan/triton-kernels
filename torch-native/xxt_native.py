import torch
torch.set_float32_matmul_precision('high')

import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def rxtx_xxt(X: torch.Tensor) -> torch.Tensor:
    orig_n, orig_m = X.shape
    
    # pad matrix dimensions to be multiples of 4
    pad_n = (4 - orig_n % 4) % 4
    pad_m = (4 - orig_m % 4) % 4
    if pad_n > 0 or pad_m > 0:
        X = torch.nn.functional.pad(X, (0, pad_m, 0, pad_n))
        
    n, m = X.shape
    n_b, m_b = n // 4, m // 4


    # extract 4x4 blocks
    X1  = X[0*n_b:1*n_b, 0*m_b:1*m_b]
    X2  = X[0*n_b:1*n_b, 1*m_b:2*m_b]
    X3  = X[0*n_b:1*n_b, 2*m_b:3*m_b]
    X4  = X[0*n_b:1*n_b, 3*m_b:4*m_b]

    X5  = X[1*n_b:2*n_b, 0*m_b:1*m_b]
    X6  = X[1*n_b:2*n_b, 1*m_b:2*m_b]
    X7  = X[1*n_b:2*n_b, 2*m_b:3*m_b]
    X8  = X[1*n_b:2*n_b, 3*m_b:4*m_b]

    X9  = X[2*n_b:3*n_b, 0*m_b:1*m_b]
    X10 = X[2*n_b:3*n_b, 1*m_b:2*m_b]
    X11 = X[2*n_b:3*n_b, 2*m_b:3*m_b]
    X12 = X[2*n_b:3*n_b, 3*m_b:4*m_b]

    X13 = X[3*n_b:4*n_b, 0*m_b:1*m_b]
    X14 = X[3*n_b:4*n_b, 1*m_b:2*m_b]
    X15 = X[3*n_b:4*n_b, 2*m_b:3*m_b]
    X16 = X[3*n_b:4*n_b, 3*m_b:4*m_b]

    # algorithm 2: 1st stage of optimized addition scheme
    y1 = X13 - X14
    y2 = X12 - X10
    w1 = X2 + X4 - X8
    w2 = X1 - X5 - X6
    w3 = X6 + X7
    w4 = X14 + X15
    w5 = y2 + X16
    w6 = X10 + X11
    w7 = X9 + y1
    w8 = X9 - X8
    w9 = X7 - X11
    w10 = X6 - X7
    w11 = X2 - X3

    # L & R matrices for the 26 multiplications
    L, R = [None]*27, [None]*27
    L[1] = -w1 + X3;            R[1] = X8 + X11
    L[2] = w2 + X7;             R[2] = X15 + X5
    L[3] = -X2 + X12;           R[3] = w5
    L[4] = X9 - X6;             R[4] = w7
    L[5] = X2 + X11;            R[5] = X15 - w3
    L[6] = X6 + X11;            R[6] = w3 - X11
    L[7] = X11;                 R[7] = w3
    L[8] = X2;                  R[8] = w3 - w4 + w5
    L[9] = X6;                  R[9] = w7 - w6 + w3
    L[10] = w1 - X3 + X7 + X11; R[10] = X11
    L[11] = X5 + w10;           R[11] = X5
    L[12] = w11 + X4;           R[12] = X8
    L[13] = -w2 + X3 - w9;      R[13] = X15
    L[14] = -w2;                R[14] = w7 + w4
    L[15] = w1;                 R[15] = w6 + w5
    L[16] = X1 - X8;            R[16] = X9 - X16
    L[17] = X12;                R[17] = -y2
    L[18] = X9;                 R[18] = y1
    L[19] = -w11;               R[19] = -X15 + X7 + X8
    L[20] = X5 + w8;            R[20] = X9
    L[21] = X8;                 R[21] = X12 + w8
    L[22] = -w10;               R[22] = X5 + w9
    L[23] = X1;                 R[23] = X13 - X5 + X16
    L[24] = -X1 + X4 + X12;     R[24] = X16
    L[25] = X9 + X2 + X10;      R[25] = X14
    L[26] = X6 + X10 + X12;     R[26] = X10

    # GPU Batched Matrix Multiplications
    L_stack = torch.stack(L[1:])
    R_stack = torch.stack(R[1:])
    
    m_stack = torch.bmm(L_stack, R_stack.transpose(1, 2))
    m = [None] + list(m_stack) 

    X_s_stack = torch.stack([X1, X2, X3, X4, X13, X14, X15, X16])
    s_stack = torch.bmm(X_s_stack, X_s_stack.transpose(1, 2))
    s1, s2, s3, s4, s5, s6, s7, s8 = s_stack

    # algorithm 3: Second stage of optimized addition scheme
    z1 = m[7] - m[11] - m[12]
    z2 = m[1] + m[12] + m[21]
    z3 = m[3] + m[17] - m[24]
    z4 = m[2] + m[11] + m[23]
    z5 = m[5] + m[7] + m[8]
    z6 = m[4] - m[18] - m[20]
    z7 = m[6] - m[7] - m[9]
    z8 = m[17] + m[18]

    C11 = s1 + s2 + s3 + s4
    C12 = m[2] - m[5] - z1 + m[13] + m[19]
    C13 = z2 + z3 + m[15] + m[16]
    C14 = z4 - z3 - z5 + m[13]
    C22 = m[1] + m[6] - z1 + m[10] + m[22]
    C23 = z2 - z6 + z7 + m[10]
    C24 = z4 + z6 + m[14] + m[16]
    C33 = m[4] - z7 - z8 + m[26]
    C34 = m[3] + z5 + z8 + m[25]
    C44 = s5 + s6 + s7 + s8

    # assembly
    C_blocks =[
        [C11,   C12,   C13,   C14],
        [C12.T, C22,   C23,   C24],[C13.T, C23.T, C33,   C34],[C14.T, C24.T, C34.T, C44]
    ]
    
    C_rows =[torch.cat(row, dim=1) for row in C_blocks]
    C = torch.cat(C_rows, dim=0)

    if pad_n > 0:
        C = C[:orig_n, :orig_n]
        
    return C


def get_error_metrics(target, pred):
    abs_diff = torch.abs(target - pred)
    max_abs = torch.max(abs_diff).item()
    max_rel = torch.max(abs_diff / (torch.abs(target) + 1e-8)).item()
    return max_abs, max_rel


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False 

    N, M = 6144, 6144 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.randn(N, M, dtype=torch.float32, device=device)
    
    print(f"Matrix shape: {X.shape} on {device}")
    
    print("\n--- Verifying Correctness in FP64 (Double Precision) ---")
    X64 = X.to(torch.float64)
    out_naive_64 = X64 @ X64.T
    out_rxtx_64 = rxtx_xxt(X64)
    
    max_abs_64, _ = get_error_metrics(out_naive_64, out_rxtx_64)
    print(f"FP64 Max Absolute Error: {max_abs_64:.10f}")
    assert max_abs_64 < 1e-7, "Mathematical mismatch detected!"
    print("RXTX is mathematically correct.")

   
    print("\n--- Checking Numerical Stability in FP32 (Single Precision) ---")
    print("Compiling function via torch.compile (kernel fusion)...")
    rxtx_compiled = torch.compile(rxtx_xxt, mode="reduce-overhead")
    
    out_naive_32 = X @ X.T
    out_rxtx_32 = rxtx_compiled(X)

    max_abs_32, max_rel_32 = get_error_metrics(out_naive_32, out_rxtx_32)
    print(f"FP32 Max Absolute Error: {max_abs_32:.4f} (Perfectly normal for fast matmul!)")
    frob_rel = torch.norm(out_naive_32 - out_rxtx_32) / torch.norm(out_naive_32)
    print(f"Frobenius relative error: {frob_rel:.4e}")

  
    def benchmark(fn, iters=50):
        for _ in range(5): fn(X)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iters):
            fn(X)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / iters

    print("\n--- Running Benchmarks (100 iterations) ---")
    torch.backends.cuda.matmul.allow_tf32 = True 
    
    naive_time = benchmark(lambda x: x @ x.T, iters=100)
    print(f"cuBLAS (X @ X.T) Average Time : {naive_time:.2f} ms")

    rxtx_time = benchmark(rxtx_compiled, iters=100)
    print(f"RXTX (Batched) Average Time   : {rxtx_time:.2f} ms")