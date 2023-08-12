// Nvcc predefines the macro __CUDACC__.
// This macro can be used in sources to test whether they are currently being compiled by nvcc.
#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#include <stdio.h>
//#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <float.h>

#include "CudaMain.cuh"
#include "ConvertFromInteger_cpu.cuh"
#include "ConvertFromInteger_curnel_cpu.cu"


template< class T>
__wchar_t* RunConvertFromIntegerKernel_cpu(void* deviceInputData, void* deviceOutputBuffer,
									 int numberOfChannels,
									 double saturation, double convertValue,
									  int widthImage, int heightImage, int strideSrcImage, int strideDstImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream)
{

	
	//dim3 blockDim(blockSizeX , blockSizeY , 1 ); // block size = number of threads
	//dim3 gridDim(gridSizeX,gridSizeY); // grid size = number of blocks
	
	convertValue = 1.0 / convertValue;

	switch (numberOfChannels)
	{
	case 1:
		{
			ConvertFromInteger_cpu<T, 1>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation, (float)convertValue,strideSrcImage,
				 strideDstImage, sizeof(T), widthImage, heightImage,
					blockSizeX, blockSizeY,
					gridSizeX, gridSizeY);
			return 0;
		}
	case 3:
		{
			ConvertFromInteger_cpu<T, 3>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation,  (float)convertValue, strideSrcImage,
				 strideDstImage, sizeof(T), widthImage, heightImage,
					blockSizeX, blockSizeY,
					gridSizeX, gridSizeY);
			return 0;
		}
	case 4:
		{
			ConvertFromInteger_cpu<T, 4>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation, (float)convertValue, strideSrcImage,
				 strideDstImage, sizeof(T), widthImage, heightImage,
					blockSizeX, blockSizeY,
					gridSizeX, gridSizeY);
			return 0;
		}
	}
	return L"Unsupported number of channels";
}

__wchar_t* RunConvertFromIntegerToFloatKernel_cpu(void* deviceInputData, void* deviceOutputBuffer,
									 int numberOfChannels,
									 double saturation, double convertValue,
									  int widthImage, int heightImage, int strideSrcImage, int strideDstImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream)
{

	
	//dim3 blockDim(blockSizeX , blockSizeY , 1 ); // block size = number of threads
	//dim3 gridDim(gridSizeX,gridSizeY); // grid size = number of blocks

	convertValue = 1.0 / convertValue;

	switch (numberOfChannels)
	{
	case 1:
		{
			ConvertFromIntegerToFloat_cpu<1>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation, (float)convertValue,strideSrcImage,
				 strideDstImage, widthImage, heightImage,
					blockSizeX, blockSizeY,
					gridSizeX, gridSizeY);
			return 0;
		}
	case 3:
		{
			ConvertFromIntegerToFloat_cpu<3>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation,  (float)convertValue, strideSrcImage,
				 strideDstImage, widthImage, heightImage,
					blockSizeX, blockSizeY,
					gridSizeX, gridSizeY);
			return 0;
		}
	case 4:
		{
			ConvertFromIntegerToFloat_cpu<4>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation, (float)convertValue, strideSrcImage,
				 strideDstImage, widthImage, heightImage,
					blockSizeX, blockSizeY,
					gridSizeX, gridSizeY);
			return 0;
		}
	}
	return L"Unsupported number of channels";
}

__wchar_t* ConvertFromIntegerToImage_Internal_cpu(void* deviceInputData, void* deviceOutputBuffer,
									 int outputSubPixelType, int numberOfChannels,
									 double saturation, double convertValue,
									  int widthImage, int heightImage, int strideSrcImage, int strideDstImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream)
{
	switch (outputSubPixelType)
	{
	case 1: // 8bit (0...0xff)
		{
			return RunConvertFromIntegerKernel_cpu<unsigned char>(deviceInputData, deviceOutputBuffer,
				numberOfChannels, saturation, convertValue,
				widthImage, heightImage, strideSrcImage, strideDstImage,
			    blockSizeX, blockSizeY, gridSizeX, gridSizeY, stream);
		}
	case 2: // 16bit (0...0xffff)
		{
			return RunConvertFromIntegerKernel_cpu<unsigned short>(deviceInputData, deviceOutputBuffer,
				numberOfChannels, saturation, convertValue,
				widthImage, heightImage, strideSrcImage, strideDstImage,
			    blockSizeX, blockSizeY, gridSizeX, gridSizeY, stream);
		}
	case 3: // Float (0...1)
		{ 
			return RunConvertFromIntegerKernel_cpu<float>(deviceInputData, deviceOutputBuffer,
				numberOfChannels, saturation, convertValue,
				widthImage, heightImage, strideSrcImage, strideDstImage,
			    blockSizeX, blockSizeY, gridSizeX, gridSizeY, stream);
		}
	}

	return L"Unsupported pixel format";
}


__wchar_t* ConvertFromIntegeToFloat__Internal_cpu(void* deviceData, 
									 int numberOfChannels, double convertValue,
									  int widthImage, int heightImage, int strideImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream)
{


	//dim3 blockDim(blockSizeX , blockSizeY , 1 ); // block size = number of threads
	//dim3 gridDim(gridSizeX,gridSizeY); // grid size = number of blocks

	convertValue = 1.0 / convertValue;

	switch (numberOfChannels)
	{
	case 1:
		{
			ConvertFromIntegerToFloat_cpu<1>
				((unsigned char*) deviceData, (float)convertValue, strideImage, widthImage, heightImage,
					blockSizeX, blockSizeY,
					gridSizeX, gridSizeY);
			return 0;
		}
	case 3:
		{
			ConvertFromIntegerToFloat_cpu<3>
				((unsigned char*) deviceData, (float)convertValue, strideImage, widthImage, heightImage,
					blockSizeX, blockSizeY,
					gridSizeX, gridSizeY);
			return 0;
		}
	case 4:
		{
			ConvertFromIntegerToFloat_cpu<4>
				((unsigned char*) deviceData, (float)convertValue, strideImage, widthImage, heightImage,
					blockSizeX, blockSizeY,
					gridSizeX, gridSizeY);
			return 0;
		}
	}
	return L"Unsupported number of channels";
}
