import torch
import triton
import triton.language as tl
import time


BLOCK_SIZE = 2048



@triton.jit
def reduce_sum_row_kernel(
    A_ptr,
    B_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    BLOCK_SIZE: tl.constexpr
):
  pid_m = tl.program_id(axis = 0)

  col_indices = tl.arange(0, BLOCK_SIZE)

  A = A_ptr + pid_m * stride_am + col_indices * stride_ak

  row_mask = (pid_m < M)
  col_mask = (col_indices < N)
  full_mask = row_mask & col_mask

  A_row = tl.load(A, mask = full_mask, other = 0.0).to(tl.float32)

  A_row_sum = tl.sum(A_row, axis = 0)

  tl.store(B_ptr + pid_m, A_row_sum.to(B_ptr.dtype.element_ty))




def reduce_sum(A: torch.Tensor):
  assert A.is_cuda and A.ndim == 2
  M, N = A.shape

  B = torch.randn((M,N), dtype = A.dtype, device = A.device)

  num_blocks = M

  reduce_sum_row_kernel[(num_blocks, )](
      A_ptr = A, B_ptr = B,
      M = M, N = N,
      stride_am = A.stride(0), stride_ak = A.stride(1),
      stride_bm = B.stride(0), stride_bk = B.stride(1),
      BLOCK_SIZE = BLOCK_SIZE
  )

  return B



if __name__ == "__main__":
  M, N = 128, 128
  A = torch.randn((M, N), dtype = torch.float32)

  A_gpu = A.cuda()

  warm_ups = 5
  for _ in range(warm_ups):
    torch.cuda.synchronize()
    start = time.time()
    output = reduce_sum(A_gpu)
    torch.cuda.synchronize()
    end = time.time()
  print("warm up done")

  total_time = 0
  num_runs = 5
  for _ in range(num_runs):
    torch.cuda.synchronize()
    start = time.time()
    output = reduce_sum(A_gpu)
    torch.cuda.synchronize()
    end = time.time()
    print((end - start) * 1000)
    total_time += (end - start) * 1000

  print("time taken: ", total_time / num_runs)

  print(output)

