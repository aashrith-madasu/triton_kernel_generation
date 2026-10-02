import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(
	a_ptr,
	b_ptr,
	output_ptr,
	element_count,
	BLOCK_SIZE: tl.constexpr,
):
	program_id = tl.program_id(axis=0)
	offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	mask = offsets < element_count

	a = tl.load(a_ptr + offsets, mask=mask)
	b = tl.load(b_ptr + offsets, mask=mask)
	tl.store(output_ptr + offsets, a + b, mask=mask)

  
def matrix_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	if not a.is_cuda or not b.is_cuda:
		raise ValueError("a and b must be CUDA tensors")
	if a.shape != b.shape:
		raise ValueError("a and b must have the same shape")
	if a.dtype != b.dtype:
		raise ValueError("a and b must have the same dtype")
	if not a.is_floating_point():
		raise ValueError("a and b must have a floating-point dtype")

	a_flat = a.contiguous().view(-1)
	b_flat = b.contiguous().view(-1)
	output = torch.empty_like(a_flat)
	element_count = output.numel()

	grid = lambda meta: (triton.cdiv(element_count, meta["BLOCK_SIZE"]),)
	matrix_add_kernel[grid](
		a_flat,
		b_flat,
		output,
		element_count,
		BLOCK_SIZE=1024,
	)
	return output.view(a.shape)


if __name__ == "__main__":
	torch.manual_seed(0)
	a = torch.randn((3, 4), device="cuda")
	b = torch.randn((3, 4), device="cuda")
	result = matrix_add(a, b)
	torch.testing.assert_close(result, a + b)
	print("Triton matrix addition verified successfully.")
