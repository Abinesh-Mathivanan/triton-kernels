import triton
import triton.language as tl
import torch


BLOCK_SIZE = 512


@triton.jit
def fused_softmax_kernel(
    A_ptr,
    B_ptr,
    M,
    N,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    BLOCK_SIZE : tl.constexpr
):
  pid_m = tl.program_id(axis = 0)

  col_indices = tl.arange(0, BLOCK_SIZE)

  x_ptrs = A_ptr + pid_m * stride_am + col_indices * stride_ak
  y_ptrs = B_ptr + pid_m * stride_bm + col_indices * stride_bk

  row_mask = (pid_m < M)
  col_mask = (col_indices < N)
  full_mask = row_mask & col_mask

  x = tl.load(x_ptrs, mask = full_mask, other = -float('inf')).to(tl.float32)

  x_max = tl.max(x, axis = 0)
  x_reduced = x - x_max
  x_exp_normalized = tl.exp(x_reduced)

  x_summed = tl.sum(x_exp_normalized, axis = 0)

  softmax_value = x_exp_normalized / x_summed

  tl.store(y_ptrs, softmax_value.to(B_ptr.dtype.element_ty), mask = full_mask)



def fused_softmax(A: torch.Tensor):
  assert A.is_cuda and A.ndim == 2

  M, N = A.shape

  B = torch.empty((M, N), dtype = torch.float32, device = A.device)

  num_blocks = M

  fused_softmax_kernel[(num_blocks, )](
      A_ptr = A, B_ptr = B,
      M = M, N = N,
      stride_am = A.stride(0), stride_ak = A.stride(1),
      stride_bm = B.stride(0), stride_bk = B.stride(1),
      BLOCK_SIZE = BLOCK_SIZE
  )

  return B



if __name__ == "__main__":
  M, N = 1024, 1024
  A = torch.randn((M, N), dtype = torch.float32)

  A_gpu = A.cuda()

  output = fused_softmax(A_gpu)

  print(*output)