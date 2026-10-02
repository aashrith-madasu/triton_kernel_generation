import torch
import triton
import triton.language as tl

@triton.jit
def gemm_swizzled_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    
    # compute 2d block ids
    pid = tl.program_id(axis=0)
    
    block_m = tl.cdiv(M, BLOCK_M)
    block_n = tl.cdiv(N, BLOCK_N)
    
    group_size = GROUP_SIZE_M * block_n
    
    group_id = pid // group_size
    pid_group = pid % group_size
    
    first_row_group = group_id * GROUP_SIZE_M
    
    safe_group_size_m = min(GROUP_SIZE_M, block_m - first_row_group)
    
    pid_m = first_row_group + pid_group % safe_group_size_m
    pid_n = pid_group // safe_group_size_m
    
    # compute row offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_id in range(tl.cdiv(K, BLOCK_K)):
        
        offs_k = k_id * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        
        # compute data pointers and masks
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        
        a = tl.load(a_ptrs, a_mask, other=0.0)
        b = tl.load(b_ptrs, b_mask, other=0.0)
        
        acc += tl.dot(a, b)
        
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    
    tl.store(c_ptrs, acc, c_mask)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    
    M, K = a.shape
    K2, N = b.shape
    
    assert K == K2, "incompatible tensor shapes"
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    GROUP_SIZE_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    gemm_swizzled_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return c



if __name__ == "__main__":
    
    for n in [128, 512, 2048, 4096, 8192]:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)
        
        c_ref = a @ b
        c_triton = gemm(a, b)
        
        print(n, torch.allclose(c_ref, c_triton, rtol=1e-2, atol=1e-1))