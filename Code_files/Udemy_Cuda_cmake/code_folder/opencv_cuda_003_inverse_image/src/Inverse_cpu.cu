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
#include "Inverse_cpu.cuh"

#include "Inverse_curnel_cpu.cu"

__wchar_t* Inverse_cpu(void* deviceInputBuffer, void* deviceOutputBuffer,
	int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,
	int widthImage, int heightImage, int strideSourceImage, int strideResultImage,
	int blockSizeX, int blockSizeY, int blockSizeZ,
	int gridSizeX, int gridSizeY)
{
	//dim3 blockDim(blockSizeX, blockSizeY, blockSizeZ); // block size = number of threads
	//dim3 gridDim(gridSizeX, gridSizeY); // grid size = number of blocks

	switch (subPixelType)
	{
	case 1: // 8bit (0...0xff)
	{
		InvertImageKernel_cpu<unsigned char>
			((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
				maxValue, alphaChannelNumber, pixelSize, channelSize,
				widthImage, heightImage, strideSourceImage, strideResultImage,
				blockSizeX, blockSizeY, blockSizeZ,
				gridSizeX, gridSizeY);
		break;
	}
	case 2: // 16bit (0...0xffff)
	{
		InvertImageKernel_cpu<unsigned short>
			((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
				maxValue, alphaChannelNumber, pixelSize, channelSize,
				widthImage, heightImage, strideSourceImage, strideResultImage,
				blockSizeX, blockSizeY, blockSizeZ,
				gridSizeX, gridSizeY);
		break;
	}
	case 4: // Float (0...1)
	{
		InvertImageKernel_cpu<float>
			((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
				(float)maxValue, alphaChannelNumber, pixelSize, channelSize,
				widthImage, heightImage, strideSourceImage, strideResultImage,
				blockSizeX, blockSizeY, blockSizeZ,
				gridSizeX, gridSizeY);
		break;
	}
	}

	return NULL;
}
