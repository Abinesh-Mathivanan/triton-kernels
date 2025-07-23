import torch 
import triton 
import triton.language as tl 


BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 256
BLOCK_SIZE_K = 32
ACCUMULATOR_DTYPE = tl.float32


@triton.jit
def gemm_batch_kernel(
    A_ptr, B_ptr, C_ptr,
    BATCH,
    M, K, N,
    stride_abatch, stride_am, stride_ak,
    stride_bbatch, stride_bm, stride_bk,
    stride_cbatch, stride_cm, stride_ck,
    BLOCK_SIZE_M = tl.constexpr,
    BLOCK_SIZE_K = tl.constexpr,
    BLOCK_SIZE_N = tl.constexpr,
    ACC_TYPE = tl.constexpr
):
  pid_m = tl.program_id(axis = 0)
  pid_n = tl.program_id(axis = 1)
  pid_batch = tl.program_id(axis = 2)

  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = ACC_TYPE)

  for k_idx in range(0, tl.cdiv(K / BLOCK_SIZE_K)):

    A_block_ptr = tl.make_block_ptr(
        A_ptr,
        shape = (BATCH, M, K),
        strides = (stride_abatch, stride_am, stride_ak),
        offsets = (pid_batch, pid_m * BLOCK_SIZE_M, k_idx * BLOCK_SIZE_K),
        block_shape = (1, BLOCK_SIZE_M, BLOCK_SIZE_K),
        order = (1, 0)
    )

    B_block_ptr = tl.make_block_ptr(
        B_ptr,
        shape = (BATCH, K, N),
        strides = (stride_bbatch, stride_bm, stride_bk),
        offsets = (pid_batch, k_idx * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_K),
        block_shape = (1, BLOCK_SIZE_K, BLOCK_SIZE_N),
        order = (1, 0)
    )

    A_tile = tl.load(A_block_ptr, boundary_check = (0, 1, 2), padding_option = "zero").squeeze(0)
    B_tile = tl.load(B_block_ptr, boundary_check = (0, 1, 2), padding_option = "zero").squeeze(0)

    accumulator += tl.dot(A_tile, B_tile)

    C_block_ptr = tl.make_block_ptr(
        C_ptr,
        shape = (BATCH, M, N),
        strides = (stride_cbatch, stride_cm, stride_ck),
        offsets = (pid_batch, pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape = (1, BLOCK_SIZE_M, BLOCK_SIZE_N),
        order = (1, 0)
    )

    tl.store(C_block_ptr, accumulator.to(C_ptr.dtype.element_ty).unsqueeze(0), boundary_check = (0, 1, 2))



def gemm_batch(A: torch.Tensor, B: torch.Tensor):
  assert A.is_cuda and B.is_cuda
  assert A.ndim == 3 and B.ndim == 3

  BATCH, M, K = A.shape
  BATCH_B, K_B, N = B.shape

  assert BATCH == BATCH_B, "batch dimension must be equal"
  assert K == K_B, "inner dimensions must be equal"

  C = torch.empty((BATCH, M, N), dtype = A.dtype, device = A.device)

  grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
  grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

  grid_BATCH = BATCH

  gemm_batch_kernel[(grid_BATCH, grid_m, grid_n)](
      A_ptr = A, B_ptr = B, C_ptr = C,
      BATCH = BATCH,
      M = M, K= K, N = N,
      stride_abatch = A.stride(0), stride_am = A.stride(1), stride_ak = A.stride(2),
      stride_bbatch = B.stride(0), stride_bm = B.stride(1), stride_bk = B.stride(2),
      stride_cbatch = C.stride(0), stride_cm = C.stride(1), stride_ck = C.stride(2),
      BLOCK_SIZE_M = BLOCK_SIZE_M,
      BLOCK_SIZE_K = BLOCK_SIZE_K,
      BLOCK_SIZE_N = BLOCK_SIZE_N,
      ACC_TYPE = ACCUMULATOR_DTYPE
  )

  return C



if __name__ == "__main__":
  BATCH_val = 16
  M_val, K_val, N_val = 512, 128, 256
  DTYPE = torch.float16

  A = torch.randn((BATCH_val, M_val, K_val), dtype = DTYPE)
  B = torch.randn((BATCH_val, K_val, N_val), dtype = DTYPE)

  A_gpu = A.cuda()
  B_gpu = B.cuda()

  output = gemm_batch(A_gpu, B_gpu)

  print(output[:10])