import triton
import triton.language as tl
import torch
import math


BLOCK_SIZE = 1024


@triton.jit
def fused_relu_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    n_elements,
    BLOCK_SIZE : tl.constexpr
):
  pid = tl.program_id(axis = 0)

  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)

  mask = offsets < n_elements

  A = tl.load(A_ptr + offsets, mask = mask)
  B = tl.load(B_ptr + offsets, mask = mask)

  add_result = A + B

  add_relu = tl.maximum(add_result, 0.0)

  tl.store(C_ptr + offsets, add_relu, mask = mask)



def fused_relu(A: torch.Tensor, B : torch.Tensor):
  assert A.is_cuda and B.is_cuda
  assert A.shape == B.shape

  n_elements = A.numel()

  C = torch.empty(n_elements, dtype = A.dtype, device = A.device)

  num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

  fused_relu_kernel[(num_blocks,)](
      A_ptr = A, B_ptr = B, C_ptr = C,
      n_elements = n_elements,
      BLOCK_SIZE = BLOCK_SIZE
  )

  return C


if __name__ == "__main__":
  size = 2**20
  A = torch.randn(size, dtype = torch.float16)
  B = torch.randn(size, dtype = torch.float16)

  A_gpu = A.cuda()
  B_gpu = B.cuda()

  output = fused_relu(A_gpu, B_gpu)

  print(output[:10])
