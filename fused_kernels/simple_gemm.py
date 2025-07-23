import torch 
import triton 
import triton.language as tl 


BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 256
BLOCK_SIZE_K = 32
ACCUMULATOR_DTYPE = tl.float32


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACC_TYPE: tl.constexpr
):
  pid_m = tl.program_id(axis = 0)
  pid_n = tl.program_id(axis = 1)

  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = ACC_TYPE)

  for idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

    A_block_ptr = tl.make_block_ptr(
        A_ptr,
        shape = (M, K),
        strides = (stride_am, stride_ak),
        offsets = (pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_K),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_K),
        order = (1, 0)
    )

    B_block_ptr = tl.make_block_ptr(
        B_ptr,
        shape = (K, N),
        strides = (stride_bm, stride_bk),
        offsets = (pid_m * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N),
        block_shape = (BLOCK_SIZE_K, BLOCK_SIZE_N),
        order = (1, 0)
    )

    A_tile = tl.load(A_block_ptr, boundary_check = (0, 1), padding_option = "zero")
    B_tile = tl.load(B_block_ptr, boundary_check = (0, 1), padding_option = "zero")

    accumulator += tl.dot(A_tile, B_tile)

    C_block_ptr = tl.make_block_ptr(
        C_ptr,
        shape = (M, N),
        strides = (stride_cm, stride_ck),
        offsets = (pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N),
        order = (1, 0)
    )

    tl.store(C_block_ptr, accumulator.to(C_ptr.dtype.element_ty), boundary_check = (0, 1))



def gemm_triton(A: torch.Tensor, B: torch.Tensor):
  assert A.is_cuda and B.is_cuda
  assert A.ndim == 2 and B.ndim == 2

  M, K = A.shape
  K_b, N = B.shape

  assert K == K_b, "inner dimensions must match"

  C = torch.empty((M, N), dtype = A.dtype, device = A.device)

  grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
  grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

  gemm_kernel[(grid_m, grid_n)](
      A_ptr = A, B_ptr = B, C_ptr = C,
      M = M, N = N, K = K,
      stride_am = A.stride(0), stride_ak = A.stride(1),
      stride_bm = B.stride(0), stride_bk = B.stride(1),
      stride_cm = C.stride(0), stride_ck = C.stride(1),
      BLOCK_SIZE_M = BLOCK_SIZE_M,
      BLOCK_SIZE_N = BLOCK_SIZE_N,
      BLOCK_SIZE_K = BLOCK_SIZE_K,
      ACC_TYPE = ACCUMULATOR_DTYPE,
  )

  return C



if __name__ == "__main__":
  M_dim, K_dim, N_dim = 1024, 512, 2048
  DTYPE = torch.float16

  A_cpu = torch.randn((M_dim, K_dim), dtype = DTYPE)
  B_cpu = torch.randn((K_dim, N_dim), dtype = DTYPE)

  A_gpu = A_cpu.cuda()
  B_gpu = B_cpu.cuda()

  output = gemm_triton(A_gpu, B_gpu)

  print(output)
