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
#include "ConvertFromInteger.cuh"
#include "ConvertFromInteger_curnel.cu"


template< class T>
__wchar_t* RunConvertFromIntegerKernel(void* deviceInputData, void* deviceOutputBuffer,
									 int numberOfChannels,
									 double saturation, double convertValue,
									  int widthImage, int heightImage, int strideSrcImage, int strideDstImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream)
{

	
	dim3 blockDim(blockSizeX , blockSizeY , 1 ); // block size = number of threads
	dim3 gridDim(gridSizeX,gridSizeY); // grid size = number of blocks
	
	convertValue = 1.0 / convertValue;

	switch (numberOfChannels)
	{
	case 1:
		{
			ConvertFromInteger<T, 1><<<gridDim ,blockDim>>>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation, (float)convertValue,strideSrcImage,
				 strideDstImage, sizeof(T), widthImage, heightImage);
			return 0;
		}
	case 3:
		{
			ConvertFromInteger<T, 3><<<gridDim ,blockDim>>>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation,  (float)convertValue, strideSrcImage,
				 strideDstImage, sizeof(T), widthImage, heightImage);
			return 0;
		}
	case 4:
		{
			ConvertFromInteger<T, 4><<<gridDim ,blockDim>>>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation, (float)convertValue, strideSrcImage,
				 strideDstImage, sizeof(T), widthImage, heightImage);
			return 0;
		}
	}
	return L"Unsupported number of channels";
}

__wchar_t* RunConvertFromIntegerToFloatKernel(void* deviceInputData, void* deviceOutputBuffer,
									 int numberOfChannels,
									 double saturation, double convertValue,
									  int widthImage, int heightImage, int strideSrcImage, int strideDstImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream)
{

	
	dim3 blockDim(blockSizeX , blockSizeY , 1 ); // block size = number of threads
	dim3 gridDim(gridSizeX,gridSizeY); // grid size = number of blocks

	convertValue = 1.0 / convertValue;

	switch (numberOfChannels)
	{
	case 1:
		{
			ConvertFromIntegerToFloat<1><<<gridDim ,blockDim>>>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation, (float)convertValue,strideSrcImage,
				 strideDstImage, widthImage, heightImage);
			return 0;
		}
	case 3:
		{
			ConvertFromIntegerToFloat<3><<<gridDim ,blockDim>>>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation,  (float)convertValue, strideSrcImage,
				 strideDstImage, widthImage, heightImage);
			return 0;
		}
	case 4:
		{
			ConvertFromIntegerToFloat<4><<<gridDim ,blockDim>>>
				((unsigned char*) deviceInputData, (unsigned char*) deviceOutputBuffer,
				  (float)saturation, (float)convertValue, strideSrcImage,
				 strideDstImage, widthImage, heightImage);
			return 0;
		}
	}
	return L"Unsupported number of channels";
}

__wchar_t* ConvertFromIntegerToImage_Internal(void* deviceInputData, void* deviceOutputBuffer,
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
			return RunConvertFromIntegerKernel<unsigned char>(deviceInputData, deviceOutputBuffer,
				numberOfChannels, saturation, convertValue,
				widthImage, heightImage, strideSrcImage, strideDstImage,
			    blockSizeX, blockSizeY, gridSizeX, gridSizeY, stream);
		}
	case 2: // 16bit (0...0xffff)
		{
			return RunConvertFromIntegerKernel<unsigned short>(deviceInputData, deviceOutputBuffer,
				numberOfChannels, saturation, convertValue,
				widthImage, heightImage, strideSrcImage, strideDstImage,
			    blockSizeX, blockSizeY, gridSizeX, gridSizeY, stream);
		}
	case 3: // Float (0...1)
		{ 
			return RunConvertFromIntegerKernel<float>(deviceInputData, deviceOutputBuffer,
				numberOfChannels, saturation, convertValue,
				widthImage, heightImage, strideSrcImage, strideDstImage,
			    blockSizeX, blockSizeY, gridSizeX, gridSizeY, stream);
		}
	}

	return L"Unsupported pixel format";
}


__wchar_t* ConvertFromIntegeToFloat__Internal(void* deviceData, 
									 int numberOfChannels, double convertValue,
									  int widthImage, int heightImage, int strideImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream)
{


	dim3 blockDim(blockSizeX , blockSizeY , 1 ); // block size = number of threads
	dim3 gridDim(gridSizeX,gridSizeY); // grid size = number of blocks

	convertValue = 1.0 / convertValue;

	switch (numberOfChannels)
	{
	case 1:
		{
			ConvertFromIntegerToFloat<1><<<gridDim ,blockDim>>>
				((unsigned char*) deviceData, (float)convertValue, strideImage, widthImage, heightImage);
			return 0;
		}
	case 3:
		{
			ConvertFromIntegerToFloat<3><<<gridDim ,blockDim>>>
				((unsigned char*) deviceData, (float)convertValue, strideImage, widthImage, heightImage);
			return 0;
		}
	case 4:
		{
			ConvertFromIntegerToFloat<4><<<gridDim ,blockDim>>>
				((unsigned char*) deviceData, (float)convertValue, strideImage, widthImage, heightImage);
			return 0;
		}
	}
	return L"Unsupported number of channels";
}
