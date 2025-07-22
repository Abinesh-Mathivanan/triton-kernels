import torch 
import triton 
import triton.language as tl 


@triton.jit
def vector_add(x_ptr, y_ptr, z_ptr, n_elements, block_size: tl.constexpr):
  pid = tl.program_id(axis=0)

  block_start = pid * block_size
  offsets = block_start + tl.arange(0, block_size)

  mask = offsets < n_elements

  x = tl.load(x_ptr + offsets, mask = mask)
  y = tl.load(y_ptr + offsets, mask = mask)

  z_val = x + y

  tl.store(z_ptr + offsets, z_val, mask = mask)



def addvector(x: torch.Tensor, y: torch.Tensor):
  assert x.is_cuda and y.is_cuda
  assert x.shape == y.shape

  n_elements = x.numel()

  output = torch.zeros_like(x)

  block_size = 1024
  num_blocks = (n_elements + block_size - 1) // block_size

  vector_add[(num_blocks, )](x, y, output, n_elements, block_size = block_size)

  return output



if __name__ == "__main__":
  size = 2**20
  x_cpu = torch.randn(size, dtype = torch.float32)
  y_cpu = torch.randn(size, dtype = torch.float32)

  x_gpu = x_cpu.cuda()
  y_gpu = y_cpu.cuda()

  output = addvector(x_gpu, y_gpu)

  print(output[:10])
