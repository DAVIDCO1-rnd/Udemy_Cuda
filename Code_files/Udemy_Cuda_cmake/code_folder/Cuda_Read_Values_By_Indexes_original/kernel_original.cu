
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

unsigned int g2dim1 = 3;
unsigned int g2dim2 = 9;

//unsigned int g2dim1 = 2576;
//unsigned int g2dim2 = 9840;

__global__ void getValsByIndexes(float2* var2, float2* g2, int2* var1, unsigned int dimension1, unsigned int dimension2, unsigned int dimension3)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	var2 = (float2*)var1;
	int max_index = dimension1 * dimension2 * dimension3;
	while (tid < max_index)
	{
		int2 currentIndexInt2 = var1[tid];
		int currentIndex = currentIndexInt2.x;
		float2 currentVal = g2[currentIndex];
		var2[tid] = currentVal;
		tid += blockDim.x * gridDim.x;
	}
}

void fill_g2_matrix(float2* &g2)
{
	int counter = 0;
	for (int i = 0; i < g2dim1; i++)
	{
		for (int j = 0; j < g2dim2; j++)
		{
			counter++;
			float xVal = counter + 10.3;
			float yVal = counter + 20.7;
			float2 currentVal;
			currentVal.x = xVal;
			currentVal.y = yVal;
			size_t currentIndex = i * g2dim2 + j;
			g2[currentIndex] = currentVal;
		}
	}
}

void fill_var1_matrix(int2* &var1, unsigned int dimension1, unsigned int dimension2, unsigned int dimension3)
{
	int counter = 0;
	for (int i = 0; i < dimension1; i++)
	{
		for (int j = 0; j < dimension2; j++)
		{
			for (int k = 0; k < dimension3; k++)
			{
				counter++;
				int xVal = counter;
				if (xVal >= g2dim1 * g2dim2)
				{
					xVal = xVal % (g2dim1 * g2dim2);
				}
				int yVal = 0;
				int2 currentVal;
				currentVal.x = xVal;
				currentVal.y = yVal;
				size_t tempIndex = j * dimension3 + k;
				size_t currentIndex = i * dimension2 * dimension3 + tempIndex;
				var1[currentIndex] = currentVal;
			}
		}
	}
}

void free_arrays(float2* g2, float2* dev_g2, int2* var1, int2* dev_var1)
{
	if (g2 != NULL)
	{
		free(g2);
	}

	if (dev_g2 != NULL)
	{
		cudaFree(dev_g2);
	}


	if (var1 != NULL)
	{
		free(var1);
	}

	if (dev_var1 != NULL)
	{
		cudaFree(dev_var1);
	}
}

void print_var2_matrix(float2* var2, unsigned int dimension1, unsigned int dimension2, unsigned int dimension3)
{
	for (int i = 0; i < dimension1; i++)
	{
		for (int j = 0; j < dimension2; j++)
		{
			for (int k = 0; k < dimension3; k++)
			{
				size_t tempIndex = j * dimension3 + k;
				size_t currentIndex = i * dimension2 * dimension3 + tempIndex;
				float2 currentVal = var2[currentIndex];
				printf("var2[%d,%d,%d] = (%f, %f)\n", i, j, k, currentVal.x, currentVal.y);
			}
			printf("\n");
		}
		printf("\n");
	}
}

int main()
{
	unsigned int dimension1 = 2;
	unsigned int dimension2 = 3;
	unsigned int dimension3 = 4;

	//unsigned int dimension1 = 664;
	//unsigned int dimension2 = 860;
	//unsigned int dimension3 = 441;

	unsigned int g2_num_of_elements = g2dim1 * g2dim2;
	size_t g2_bytes = g2_num_of_elements * sizeof(float2);
	float2* g2 = (float2*)malloc(g2_bytes);
	fill_g2_matrix(g2);

	size_t var1_num_of_elements = dimension1 * dimension2 * dimension3;
	size_t var1_bytes = var1_num_of_elements * sizeof(int2);
	int2* var1 = (int2*)malloc(var1_bytes);

	fill_var1_matrix(var1, dimension1, dimension2, dimension3);



	float2* dev_g2 = NULL;
	int2* dev_var1 = NULL;
	cudaError_t cudaStatus_g2_alloc = cudaMalloc((void**)&dev_g2, g2_bytes);
	if (cudaStatus_g2_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for g2 failed!");
		free_arrays(g2, dev_g2, var1, dev_var1);
		return 1;
	}

	cudaError_t cudaStatus_var1_alloc = cudaMalloc((void**)&dev_var1, var1_bytes);
	if (cudaStatus_var1_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for var1 failed!");
		free_arrays(g2, dev_g2, var1, dev_var1);
		return 1;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_g2_memcpy = cudaMemcpy(dev_g2, g2, g2_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_g2_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy g2 failed!");
		free_arrays(g2, dev_g2, var1, dev_var1);
		return 1;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_var1_memcpy = cudaMemcpy(dev_var1, var1, var1_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_var1_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy var1 failed!");
		free_arrays(g2, dev_g2, var1, dev_var1);
		return 1;
	}

	float2* dev_var2 = (float2*)dev_var1;
	float2* var2 = (float2*)var1;
	size_t var2_bytes = var1_bytes;

	getValsByIndexes << <12, 2 >> > (dev_var2, dev_g2, dev_var1, dimension1, dimension2, dimension3);

	// Check for any errors launching the kernel
	cudaError_t cudaStatusLastError = cudaGetLastError();
	if (cudaStatusLastError != cudaSuccess) {
		fprintf(stderr, "getValsByIndexes launch failed: %s\n", cudaGetErrorString(cudaStatusLastError));
		free_arrays(g2, dev_g2, var1, dev_var1);
		return 1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaError_t cudaStatusDeviceSynchronize = cudaDeviceSynchronize();
	if (cudaStatusDeviceSynchronize != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getValsByIndexes!\n", cudaStatusDeviceSynchronize);
		free_arrays(g2, dev_g2, var1, dev_var1);
		return 1;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaError_t cudaStatus_var2_memcpy = cudaMemcpy(var2, dev_var2, var2_bytes, cudaMemcpyDeviceToHost);
	if (cudaStatus_var2_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy var2 failed!");
		free_arrays(g2, dev_g2, var1, dev_var1);
		return 1;
	}

	print_var2_matrix(var2, dimension1, dimension2, dimension3);


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatusDeviceReset = cudaDeviceReset();
	if (cudaStatusDeviceReset != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		free_arrays(g2, dev_g2, var1, dev_var1);
		return 1;
	}

	free_arrays(g2, dev_g2, var1, dev_var1);
	return 0;
}