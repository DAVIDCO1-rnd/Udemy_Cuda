// Demonstration of kernel execution configuration for a one-dimensional
// grid.
// Example for video 2.2.

#include <cuda_runtime_api.h>
#include <stdio.h>

// Error checking macro
#define cudaCheckError(code)                                             \
  {                                                                      \
    if ((code) != cudaSuccess) {                                         \
      fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
              cudaGetErrorString(code));                                 \
    }                                                                    \
  }

__global__ void kernel_1d()
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  printf("blockIdx.x %d, threadIdx.x %d, index_x %d\n", blockIdx.x, threadIdx.x, index_x);
  printf("blockIdx.y %d, threadIdx.y %d, index_y %d\n", blockIdx.y, threadIdx.y, index_y);
  printf("\n");
}

int main()
{
  //kernel_1d<<<4, 64>>>();
  //cudaCheckError(cudaDeviceSynchronize());

	dim3 block_dim(2, 16);
	dim3 grid_dim(2, 2);
	kernel_1d<<<grid_dim, block_dim >>>();
	cudaCheckError(cudaDeviceSynchronize());
}
