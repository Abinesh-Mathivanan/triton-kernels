import torch 
import triton 
import triton.language as tl 

@triton.jit
def multiply_vector(x_ptr, y_ptr, output, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis = 0)

  block_start =  pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)

  mask = offsets < n_elements

  x = tl.load(x_ptr + offsets, mask = mask)
  y = tl.load(y_ptr + offsets, mask = mask)

  output_val = x * y

  tl.store(output + offsets, output_val, mask = mask)



def vecmul(x: torch.Tensor, y: torch.Tensor):
  assert x.is_cuda and y.is_cuda
  assert x.shape == y.shape

  n_elements = x.numel()

  output = torch.zeros_like(x)

  BLOCK_SIZE = 1024

  num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

  multiply_vector[(num_blocks, )](x, y, output, n_elements, BLOCK_SIZE = BLOCK_SIZE)

  return output



if __name__ == "__main__":
  size = 2**20
  x = torch.randn(size, dtype = torch.float32)
  y = torch.randn(size, dtype = torch.float32)

  x_gpu = x.cuda()
  y_gpu = y.cuda()

  output = vecmul(x_gpu, y_gpu)

  torch_output = x * y

  print(output[:10])
  print(torch_output[:10])

