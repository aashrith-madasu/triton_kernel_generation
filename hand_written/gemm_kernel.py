import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        
	pid_m = tl.program_id(0)
	pid_n = tl.program_id(1)

	offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_k = tl.arange(0, BLOCK_K)

	a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
	b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

	acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

	num_tiles_k = tl.cdiv(K, BLOCK_K)

	for k_i in range(num_tiles_k):
		k_rem = K - k_i * BLOCK_K

		a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
		b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)

		a = tl.load(a_ptrs, a_mask, other=0.0)
		b = tl.load(b_ptrs, b_mask, other=0.0)

		acc += tl.dot(a, b)

		a_ptrs += BLOCK_K * stride_ak
		b_ptrs += BLOCK_K * stride_bk

	
	c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
	c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
	
	tl.store(c_ptrs, acc.to(tl.float16), c_mask)




def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    
    assert a.shape[1] == b.shape[0], "incompatible tensor shapes"
    
    M, K = a.shape
    _, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return c


if __name__ == "__main__":
    
    for n in [128, 512, 2048, 4096, 8192]:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)
        
        c_ref = a @ b
        c_triton = gemm(a, b)
        
        print(n, torch.allclose(c_ref, c_triton))
        # print(n, triton.testing.do_bench(lambda: a @ b), triton.testing.do_bench(lambda: gemm(a, b)))
    
    
    