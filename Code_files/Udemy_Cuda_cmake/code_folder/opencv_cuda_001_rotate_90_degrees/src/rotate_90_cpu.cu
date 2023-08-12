// Nvcc predefines the macro __CUDACC__.
// This macro can be used in sources to test whether they are currently being compiled by nvcc.
#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#include <stdio.h>
//#include <cutil.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>

#include "CudaMain.cuh"
#include "rotate_90_cpu.cuh"

#include "rotate_90_curnel_cpu.cu"

__wchar_t* rotate_90_cpu(void* deviceInputBuffer, void* deviceOutputBuffer, int subPixelType,
	int widthImage, int heightImage, int is_clockwise,
	int blockSizeX, int blockSizeY, int blockSizeZ,
	int gridSizeX, int gridSizeY)
{
	//dim3 blockDim(blockSizeX, blockSizeY, blockSizeZ); // block size = number of threads
	//dim3 gridDim(gridSizeX, gridSizeY); // grid size = number of blocks

	int pixel_size = subPixelType;
	switch (subPixelType)
	{
	case 1: // 8bit (0...0xff)
	{
		rotate_90_kernel_cpu<unsigned char>
			((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
				widthImage, heightImage, pixel_size, is_clockwise,
				blockSizeX, blockSizeY, blockSizeZ,
				gridSizeX, gridSizeY);
		break;
	}
	case 2: // 16bit (0...0xffff)
	{
		rotate_90_kernel_cpu<unsigned short>
			((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
				widthImage, heightImage, pixel_size, is_clockwise,
				blockSizeX, blockSizeY, blockSizeZ,
				gridSizeX, gridSizeY);
		break;
	}
	case 4: // Float (0...1)
	{
		rotate_90_kernel_cpu<float>
			((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
				widthImage, heightImage, pixel_size, is_clockwise,
				blockSizeX, blockSizeY, blockSizeZ,
				gridSizeX, gridSizeY);
		break;
	}
	}

	return NULL;
}
