#include "fake_cuda_in_cpu.cuh"

void syncthreads_cpu()
{
	//do nothing
	return;
}

void atomicAdd_cpu(int* ptr, int val_to_add)
{
	*ptr = *ptr + val_to_add;
}