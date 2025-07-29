import torch
import triton
import triton.language as tl



@triton.jit
def gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr,
    ACCUMULATOR : tl.constexpr,
):
  pid_m = tl.program_id(axis = 0)
  pid_n = tl.program_id(axis = 1)

  accumulator = tl.zeroes((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = ACCUMULATOR)

  for k_idx in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    A_block_ptr = tl.make_block_ptr(
        A_ptr,
        shape = (M, K),
        strides = (stride_am, stride_ak),
        offsets = (pid_m * BLOCK_SIZE_M, k_idx * BLOCK_SIZE_K),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_K),
        order = (1, 0)
    )

    B_block_ptr = tl.make_block_ptr(
        B_ptr,
        shape = (K, N),
        strides = (stride_bm, stride_bk),
        offsets = (k_idx * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N),
        block_shape = (BLOCK_SIZE_K, BLOCK_SIZE_N),
        order = (1, 0)
    )

    A_tile = tl.load(A_block_ptr, boundary_check = (0, 1), padding_mode = "zero")
    B_tile = tl.load(B_block_ptr, boundary_check = (0, 1), padding_mode = "zero")

    accumulator += tl.dot(A_tile, B_tile)

  C_block_ptr = tl.make_block_ptr(
      C_ptr,
      shape = (M, N),
      strides = (stride_cm, stride_ck),
      offsets = (pid_m * BLOCK_SIZE_M, pid * BLOCK_SIZE_N),
      block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N),
      order = (1, 0)
  )

  tl.store(C_block_ptr, accumulator.to(C_ptr.dtype.element_ty), boundary_check = (0, 1))



@triton.jit
def gemm_b_trans_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr,
    ACCUMULATOR : tl.constexpr,
):
  pid_m = tl.program_id(axis = 0)
  pid_n = tl.program_id(axis = 1)

  accumulator = tl.zeroes((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = ACCUMULATOR)

  for k_idx in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    A_block_ptr = tl.make_block_ptr(
        A_ptr,
        shape = (M, K),
        strides = (stride_am, stride_ak),
        offsets = (pid_m * BLOCK_SIZE_M, k_idx * BLOCK_SIZE_K),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_K),
        order = (1, 0)
    )

    B_block_ptr = tl.make_block_ptr(
        B_ptr,
        shape = (N, K),
        strides = (stride_bn, stride_bk),
        offsets = (pid_n * BLOCK_SIZE_N, k_idx * BLOCK_SIZE_K),
        block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_K),
        order = (1, 0)
    )

    A_tile = tl.load(A_block_ptr, boundary_check = (0, 1), padding_mode = "zero")
    B_tile = tl.load(B_block_ptr, boundary_check = (0, 1), padding_mode = "zero")
    B_tile_transposed = B_tile.transpose(0, 1)

    accumulator += tl.dot(A_tile, B_tile_transposed)

  C_block_ptr = tl.make_block_ptr(
      C_ptr,
      shape = (M, N),
      strides = (stride_cm, stride_ck),
      offsets = (pid_m * BLOCK_SIZE_M, pid * BLOCK_SIZE_N),
      block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N),
      order = (1, 0)
  )

  tl.store(C_block_ptr, accumulator.to(C_ptr.dtype.element_ty), boundary_check = (0, 1))



@triton.jit
def softmax_kernel(
    A_ptr,
    B_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    BLOCK_SIZE : tl.constexpr,
):
  pid_m = tl.program_id(axis = 0)

  current_max_val = tl.zeros([1], dtype=tl.float32) - float('inf')
  current_l_val = tl.zeros([1], dtype=tl.float32)

  for start_n_block in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
      col_indices_in_chunk = (start_n_block * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
      A_chunk_ptrs = A_ptr + pid_m * stride_am + col_indices_in_chunk * stride_ak

      row_bounds_mask = pid_m < M
      col_bounds_mask = col_indices_in_chunk < N
      full_chunk_mask = row_bounds_mask & col_bounds_mask

      a_chunk = tl.load(A_chunk_ptrs, mask=full_chunk_mask, other=-float('inf')).to(tl.float32)

      chunk_max = tl.max(a_chunk, axis=0)
      chunk_sum_exp = tl.sum(tl.exp(a_chunk - chunk_max), axis=0)

      new_max_val = tl.maximum(current_max_val, chunk_max)
      alpha = tl.exp(current_max_val - new_max_val)
      current_l_val = current_l_val * alpha
      current_l_val = current_l_val + tl.exp(chunk_max - new_max_val) * chunk_sum_exp
      current_max_val = new_max_val

  for start_n_block in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
      col_indices_in_chunk = (start_n_block * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
      A_chunk_ptrs = A_ptr + pid_m * stride_am + col_indices_in_chunk * stride_ak
      C_chunk_ptrs = C_ptr + pid_m * stride_cm + col_indices_in_chunk * stride_ck

      row_bounds_mask = pid_m < M
      col_bounds_mask = col_indices_in_chunk < N
      full_chunk_mask = row_bounds_mask & col_bounds_mask

      a_chunk = tl.load(A_chunk_ptrs, mask=full_chunk_mask, other=-float('inf')).to(tl.float32)

      shifted_scores = a_chunk - current_max_val
      exp_scores = tl.exp(shifted_scores)
      probabilities_chunk = exp_scores / current_l_val

      tl.store(C_chunk_ptrs, probabilities_chunk.to(C_ptr.dtype.element_ty), mask=full_chunk_mask)



def vanilla_attention_block(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4
    assert Q.shape == K.shape == V.shape

    BATCH, NUM_HEADS, N_CTX, D_HEAD = Q.shape

    scores = torch.empty((BATCH, NUM_HEADS, N_CTX, N_CTX), dtype=torch.float32, device=Q.device)

    q_slice = Q[0, 0, :, :]
    k_slice = K[0, 0, :, :]
    v_slice = V[0, 0, :, :]

    qk_t_scores = torch.empty((N_CTX, N_CTX), dtype=torch.float32, device=Q.device)

    M_scores, K_scores = q_slice.shape
    K_scores_B, N_scores_B = k_slice.shape
    scale = D_HEAD ** -0.5

    gemm_b_trans_kernel[(triton.cdiv(M_scores, BLOCK_SIZE_M_GEMM), triton.cdiv(N_scores_B, BLOCK_SIZE_N_GEMM))](
        A_ptr=q_slice, B_ptr=k_slice, C_ptr=qk_t_scores,
        M=M_scores, N=N_scores_B, K=K_scores,
        stride_am=q_slice.stride(0), stride_ak=q_slice.stride(1),
        stride_bn=k_slice.stride(0), stride_bk=k_slice.stride(1),
        stride_cm=qk_t_scores.stride(0), stride_ck=qk_t_scores.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M_GEMM,
        BLOCK_SIZE_N=BLOCK_SIZE_N_GEMM,
        BLOCK_SIZE_K=BLOCK_SIZE_K_GEMM,
        ACC_TYPE=ACCUMULATOR_DTYPE_GEMM,
    )

    qk_t_scores *= scale

    if is_causal:
        mask = torch.triu(torch.ones(N_CTX, N_CTX, device=Q.device), diagonal=1).bool()
        qk_t_scores.masked_fill_(mask, float('-inf'))

    attention_probs = torch.empty_like(qk_t_scores)

    softmax_kernel_robust[(N_CTX,)](
        A_ptr=qk_t_scores, C_ptr=attention_probs,
        M=N_CTX, N=N_CTX,
        stride_am=qk_t_scores.stride(0), stride_ak=qk_t_scores.stride(1),
        stride_cm=attention_probs.stride(0), stride_ck=attention_probs.stride(1),
    )

    output_slice = torch.empty((N_CTX, D_HEAD), dtype=Q.dtype, device=Q.device)

    M_out, K_out = attention_probs.shape
    K_out_B, N_out_B = v_slice.shape

    gemm_kernel[(triton.cdiv(M_out, BLOCK_SIZE_M_GEMM), triton.cdiv(N_out_B, BLOCK_SIZE_N_GEMM))](
        A_ptr=attention_probs, B_ptr=v_slice, C_ptr=output_slice,
        M=M_out, N=N_out_B, K=K_out,
        stride_am=attention_probs.stride(0), stride_ak=attention_probs.stride(1),
        stride_bm=v_slice.stride(0), stride_bk=v_slice.stride(1),
        stride_cm=output_slice.stride(0), stride_ck=output_slice.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M_GEMM,
        BLOCK_SIZE_N=BLOCK_SIZE_N_GEMM,
        BLOCK_SIZE_K=BLOCK_SIZE_K_GEMM,
        ACC_TYPE=ACCUMULATOR_DTYPE_GEMM,
    )

    return output_slice


def multi_head_attention_triton(Q, K, V, is_causal=False):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    BATCH, NUM_HEADS, N_CTX, D_HEAD = Q.shape

    output = torch.empty(Q, dtype=Q.dtype, device=Q.device)

    for batch_idx in range(BATCH):
        for head_idx in range(NUM_HEADS):
            q_slice = Q[batch_idx, head_idx, :, :]
            k_slice = K[batch_idx, head_idx, :, :]
            v_slice = V[batch_idx, head_idx, :, :]
            output_slice = output[batch_idx, head_idx, :, :]

            output_slice = vanilla_attention_block(q_slice, k_slice, v_slice, is_causal)

    return output



if __name__ == "__main__":
  BATCH, NUM_HEADS, N_CTX, D_HEAD = 4, 12, 1024, 64

  DTYPE = torch.float16
  IS_CAUSAL = True

  Q_cpu = torch.randn((BATCH, NUM_HEADS, N_CTX, D_HEAD), dtype=DTYPE)
  K_cpu = torch.randn((BATCH, NUM_HEADS, N_CTX, D_HEAD), dtype=DTYPE)
  V_cpu = torch.randn((BATCH, NUM_HEADS, N_CTX, D_HEAD), dtype=DTYPE)

  Q_gpu, K_gpu, V_gpu = Q_cpu.cuda(), K_cpu.cuda(), V_cpu.cuda()

  triton_multi_head_attention = multi_head_attention_triton(Q_gpu, K_gpu, V_gpu, IS_CAUSAL)

  print(triton_multi_head_attention)