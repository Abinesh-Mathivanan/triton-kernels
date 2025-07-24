import torch
import triton
import triton.language as tl


BLOCK_SIZE = 1024 


@triton.jit
def fused_silu_kernel(
    A_ptr,
    B_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
  pid_m = tl.program_id(axis = 0)

  block_start = pid_m * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements

  A = tl.load(A_ptr + offsets, mask = mask)

  neg_exp = tl.exp(-A)
  denominator = 1 + neg_exp
  reduced_silu = 1 / (1 + neg_exp)
  scaled_silu = A * reduced_silu

  tl.store(B_ptr + offsets, scaled_silu.to(B_ptr.dtype.element_ty), mask = mask)



def fused_silu(A: torch.Tensor):
  assert A.is_cuda and A.ndim == 1

  n_elements = A.numel()

  B = torch.randn(n_elements, dtype = A.dtype, device = A.device)

  num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

  fused_silu_kernel[(num_blocks, )](
      A_ptr = A, B_ptr = B,
      n_elements = n_elements,
      BLOCK_SIZE = BLOCK_SIZE
  )

  return B



if __name__ == "__main__":
  size = 2**20
  A = torch.randn(size, dtype = torch.float32)

  A_gpu = A.cuda()

  output = fused_silu(A_gpu)

  print(output)
