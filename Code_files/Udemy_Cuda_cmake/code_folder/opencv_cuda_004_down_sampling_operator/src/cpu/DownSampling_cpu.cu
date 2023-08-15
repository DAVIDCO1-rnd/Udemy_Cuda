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
#include "ConvertFromInteger_cpu.cuh"
#include "CudaMain.cuh"
#include "DownSampling_cpu.cuh"
#include "DownSampling_curnel_cpu.cu"


// Support DS for segentation (NO billinear)

__wchar_t* DownSampleTopLeft_cpu(
	void* deviceInputBuffer, void* deviceOutputBuffer,
	int widthSourceImage, int heightSourceImage, int strideSourceImage,
	int widthDestImage, int heightDestImage, int strideDestImage,
	double horizontalScale, double verticalScale,
	int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,
	int blockSizeX, int blockSizeY, int blockSizeZ, int gridSizeX, int gridSizeY)
{
	return DownSampleTopLeft_Parallel_cpu(deviceInputBuffer, deviceOutputBuffer, widthSourceImage, heightSourceImage,
		strideSourceImage, widthDestImage, heightDestImage, strideDestImage, horizontalScale, verticalScale,
		subPixelType, maxValue, alphaChannelNumber, pixelSize, channelSize,
		blockSizeX, blockSizeY, blockSizeZ, gridSizeX, gridSizeY, 0);
}

__wchar_t* DownSampleTopLeft_Parallel_cpu(
	void* deviceInputBuffer, void* deviceOutputBuffer,
	int widthSourceImage, int heightSourceImage, int strideSourceImage,
	int widthDestImage, int heightDestImage, int strideDestImage,
	double horizontalScale, double verticalScale,
	int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,
	int blockSizeX, int blockSizeY, int blockSizeZ, int gridSizeX, int gridSizeY, void* stream)
{


	//dim3 blockDim(blockSizeX, blockSizeY, blockSizeZ); // block size = number of threads
	//dim3 gridDim(gridSizeX, gridSizeY); // grid size = number of blocks



	switch (subPixelType)
	{
	case 1: // 8bit (0...0xff)
	{

		DownSampleTopLeftKernel_cpu<unsigned char>
			((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
			widthSourceImage, heightSourceImage, strideSourceImage,
			widthDestImage, heightDestImage, strideDestImage,
			(float)(1.0 / horizontalScale), (float)(1.0 / verticalScale),
			maxValue, pixelSize, channelSize,
			blockSizeX, blockSizeY, blockSizeZ,
			gridSizeX, gridSizeY
			);
		break;

	}
	case 2: // 16bit (0...0xffff)
	{

		DownSampleTopLeftKernel_cpu<unsigned short>
			((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
			widthSourceImage, heightSourceImage, strideSourceImage,
			widthDestImage, heightDestImage, strideDestImage,
			(float)(1.0 / horizontalScale), (float)(1.0 / verticalScale),
			maxValue, pixelSize, channelSize,
			blockSizeX, blockSizeY, blockSizeZ,
			gridSizeX, gridSizeY
			);
		break;

	}
	case 4: // Float (0...1)
	{

		DownSampleTopLeftKernelFloat_cpu<float>
			((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
			widthSourceImage, heightSourceImage, strideSourceImage,
			widthDestImage, heightDestImage, strideDestImage,
			(float)(1.0 / horizontalScale), (float)(1.0 / verticalScale),
			pixelSize, channelSize,
			blockSizeX, blockSizeY, blockSizeZ,
			gridSizeX, gridSizeY
			);
		break;
	}
	}

	return 0;
}



// ------------------------------------------------------------------------------------------------------------------


__wchar_t* DownSample_cpu(
		void* deviceInputBuffer, void* deviceOutputBuffer,
		int widthSourceImage, int heightSourceImage, int strideSourceImage,
		int widthDestImage, int heightDestImage, int strideDestImage,
		double horizontalScale, double verticalScale,
		int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,			
		int blockSizeX, int blockSizeY, int blockSizeZ, int gridSizeX, int gridSizeY)
{
	return DownSample_Parallel_cpu(deviceInputBuffer, deviceOutputBuffer, widthSourceImage, heightSourceImage,
		strideSourceImage, widthDestImage, heightDestImage, strideDestImage, horizontalScale, verticalScale,
		subPixelType, maxValue, alphaChannelNumber, pixelSize, channelSize,			
		blockSizeX, blockSizeY, blockSizeZ, gridSizeX, gridSizeY, 0); 
}

__wchar_t* DownSample_Parallel_cpu(
		void* deviceInputBuffer, void* deviceOutputBuffer,
		int widthSourceImage, int heightSourceImage, int strideSourceImage,
		int widthDestImage, int heightDestImage, int strideDestImage,
		double horizontalScale, double verticalScale,
		int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,			
		int blockSizeX, int blockSizeY, int blockSizeZ, int gridSizeX, int gridSizeY, void* stream)
{


	//dim3 blockDim(blockSizeX, blockSizeY, blockSizeZ); // block size = number of threads
 //   dim3 gridDim(gridSizeX, gridSizeY); // grid size = number of blocks

	switch (subPixelType)
	{
		case 1: // 8bit (0...0xff)
		{
		
			DownSampleKernel_cpu<unsigned char>
			  ((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
				widthSourceImage, heightSourceImage, strideSourceImage,
				widthDestImage, heightDestImage, strideDestImage,
				(float)(1.0 / horizontalScale), (float)(1.0 / verticalScale),
				maxValue, pixelSize, channelSize,
				blockSizeX, blockSizeY, blockSizeZ,
				gridSizeX, gridSizeY
				);
			break;
			
		}
		case 2: // 16bit (0...0xffff)
		{
		
			DownSampleKernel_cpu<unsigned short>
			  ((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
				widthSourceImage, heightSourceImage, strideSourceImage,
				widthDestImage, heightDestImage, strideDestImage,
				(float)(1.0 / horizontalScale), (float)(1.0 / verticalScale),
				maxValue, pixelSize, channelSize,
				blockSizeX, blockSizeY, blockSizeZ,
				gridSizeX, gridSizeY
				);
			break;
		
		}
		case 4: // Float (0...1)
		{
			
			DownSampleKernelFloat_cpu<float>
			  ((unsigned char*)deviceInputBuffer, (unsigned char*)deviceOutputBuffer,
				widthSourceImage, heightSourceImage, strideSourceImage,
				widthDestImage, heightDestImage, strideDestImage,
				(float)(1.0 / horizontalScale), (float)(1.0 / verticalScale),
				pixelSize, channelSize,
				blockSizeX, blockSizeY, blockSizeZ,
				gridSizeX, gridSizeY
				);
			break;
		}
	}
	
	return 0;
}


template< class T>
__wchar_t* RunFastDownSampleKernel_cpu(void* deviceInputBuffer, void* deviceIntegerOutputBuffer, 
								   void* deviceOutputBuffer, double maxValue, double convertValue, 
								   int numberOfChannels, int subPixelType, 
								   int widthSourceImage, int heightSourceImage, int strideSourceImage,
								   int widthDestImage, int heightDestImage, int strideIntegerDestImage, int strideDestImage,
								   double horizontalScale, double verticalScale,
								   int blockSizeSrcX, int blockSizeSrcY, int gridSizeSrcX, int gridSizeSrcY, 
								   int blockSizeDstX, int blockSizeDstY, int gridSizeDstX, int gridSizeDstY,
								   void* stream)
{



	//dim3 blockDim(blockSizeSrcX, blockSizeSrcY, 1); // block size = number of threads
 //   dim3 gridDim(gridSizeSrcX, gridSizeSrcY); // grid size = number of blocks

	float downSampleConvertValue = (float)(convertValue * horizontalScale * verticalScale);

	switch (numberOfChannels)
	{
	case 1:
		{
			if (horizontalScale == 1)
			{
				FastDownSampleVerticalKernel_cpu<T, 1, sizeof(T) >
					((unsigned char*)deviceInputBuffer, (unsigned char*)deviceIntegerOutputBuffer,
					widthSourceImage, heightSourceImage, strideSourceImage,
					 strideIntegerDestImage,
					(float)verticalScale, downSampleConvertValue,
					blockSizeSrcX, blockSizeSrcY,
					gridSizeSrcX, gridSizeSrcY);
			}
			else if (verticalScale == 1)
			{
				FastDownSampleHorizontalKernel_cpu<T, 1, sizeof(T)>
					((unsigned char*)deviceInputBuffer, (unsigned char*)deviceIntegerOutputBuffer,
					widthSourceImage, heightSourceImage, strideSourceImage, 
					 strideIntegerDestImage,
					(float)horizontalScale, downSampleConvertValue,
					blockSizeSrcX, blockSizeSrcY,
					gridSizeSrcX, gridSizeSrcY);
			}
			else
			{
				FastDownSampleKernel_cpu<T, 1, sizeof(T)>
					((unsigned char*)deviceInputBuffer, (unsigned char*)deviceIntegerOutputBuffer,
					widthSourceImage, heightSourceImage, strideSourceImage,
					strideIntegerDestImage,
					(float)horizontalScale, (float)verticalScale, downSampleConvertValue,
					blockSizeSrcX, blockSizeSrcY,
					gridSizeSrcX, gridSizeSrcY);
			}
			break;
		}
	case 3:
		{
			if (horizontalScale == 1)
			{
				FastDownSampleVerticalKernel_cpu<T, 3, sizeof(T)>
					((unsigned char*)deviceInputBuffer, (unsigned char*)deviceIntegerOutputBuffer,
					widthSourceImage, heightSourceImage, strideSourceImage,
					strideIntegerDestImage,
					(float)verticalScale, downSampleConvertValue,
					blockSizeSrcX, blockSizeSrcY,
					gridSizeSrcX, gridSizeSrcY);
			}
			else if (verticalScale == 1)
			{
				FastDownSampleHorizontalKernel_cpu<T, 3, sizeof(T)>
					((unsigned char*)deviceInputBuffer, (unsigned char*)deviceIntegerOutputBuffer,
					widthSourceImage, heightSourceImage, strideSourceImage, 
					strideIntegerDestImage,
					(float)horizontalScale, downSampleConvertValue,
					blockSizeSrcX, blockSizeSrcY,
					gridSizeSrcX, gridSizeSrcY);
			}
			else
			{
				FastDownSampleKernel_cpu<T, 3, sizeof(T)>
					((unsigned char*)deviceInputBuffer, (unsigned char*)deviceIntegerOutputBuffer,
					widthSourceImage,  heightSourceImage,  strideSourceImage, 
					strideIntegerDestImage,
					(float)horizontalScale, (float)verticalScale, downSampleConvertValue,
					blockSizeSrcX, blockSizeSrcY,
					gridSizeSrcX, gridSizeSrcY);
			}

			break;
		}
	case 4:
		{
			if (horizontalScale == 1)
			{
				FastDownSampleVerticalKernel_cpu<T, 4, sizeof(T)>
					((unsigned char*)deviceInputBuffer, (unsigned char*)deviceIntegerOutputBuffer,
					widthSourceImage, heightSourceImage, strideSourceImage, 
					 strideIntegerDestImage,
					(float)verticalScale, downSampleConvertValue,
					blockSizeSrcX, blockSizeSrcY,
					gridSizeSrcX, gridSizeSrcY);
			}
			else if (verticalScale == 1)
			{
				FastDownSampleHorizontalKernel_cpu<T, 4, sizeof(T)>
					((unsigned char*)deviceInputBuffer, (unsigned char*)deviceIntegerOutputBuffer,
					widthSourceImage, heightSourceImage, strideSourceImage, 
					strideIntegerDestImage,
					(float)horizontalScale, downSampleConvertValue,
					blockSizeSrcX, blockSizeSrcY,
					gridSizeSrcX, gridSizeSrcY);
			}
			else
			{
				FastDownSampleKernel_cpu<T, 4, sizeof(T)>
					((unsigned char*)deviceInputBuffer, (unsigned char*)deviceIntegerOutputBuffer,
					widthSourceImage,  heightSourceImage,  strideSourceImage, 
					strideIntegerDestImage,
					(float)horizontalScale, (float)verticalScale, downSampleConvertValue,
					blockSizeSrcX, blockSizeSrcY,
					gridSizeSrcX, gridSizeSrcY);
			}

			break;
		}
	default:
		return L"Unsupported number of channels";
	}

	if (subPixelType == 4)
	{
		return RunConvertFromIntegerToFloatKernel_cpu(deviceIntegerOutputBuffer, deviceOutputBuffer,
				 numberOfChannels, maxValue, convertValue,
				widthDestImage,  heightDestImage,  strideIntegerDestImage, strideDestImage,
				blockSizeDstX, blockSizeDstY, gridSizeDstX,  gridSizeDstY, stream);
	}

	return RunConvertFromIntegerKernel_cpu<T>(deviceIntegerOutputBuffer, deviceOutputBuffer,
		numberOfChannels, maxValue, convertValue,
		widthDestImage,  heightDestImage,  strideIntegerDestImage, strideDestImage,
		blockSizeDstX, blockSizeDstY, gridSizeDstX,  gridSizeDstY, stream);

}

__wchar_t* FastDownSample_cpu(void* deviceInputBuffer, void* deviceIntegerOutputBuffer, 
								   void* deviceOutputBuffer, double maxValue, double convertValue, 
								   int numberOfChannels, int subPixelType, 
								   int widthSourceImage, int heightSourceImage, int strideSourceImage,
								   int widthDestImage, int heightDestImage, int strideIntegerDestImage, int strideDestImage,
								   double horizontalScale, double verticalScale,
								   int blockSizeSrcX, int blockSizeSrcY, int gridSizeSrcX, int gridSizeSrcY, 		
								   int blockSizeDstX, int blockSizeDstY, int gridSizeDstX, int gridSizeDstY)
{
	return FastDownSample_Parallel_cpu(deviceInputBuffer, deviceIntegerOutputBuffer, deviceOutputBuffer, 
		maxValue, convertValue, numberOfChannels,  subPixelType, 
		widthSourceImage, heightSourceImage, strideSourceImage,
		widthDestImage, heightDestImage, strideIntegerDestImage, strideDestImage,  horizontalScale,  verticalScale,
		blockSizeSrcX, blockSizeSrcY, gridSizeSrcX, gridSizeSrcY, 
		blockSizeDstX, blockSizeDstY, gridSizeDstX,  gridSizeDstY, 0);
}

__wchar_t* FastDownSample_Parallel_cpu(void* deviceInputBuffer, void* deviceIntegerOutputBuffer, 
								   void* deviceOutputBuffer, double maxValue, double convertValue, 
								   int numberOfChannels, int subPixelType, 
								   int widthSourceImage, int heightSourceImage, int strideSourceImage,
								   int widthDestImage, int heightDestImage, int strideIntegerDestImage, int strideDestImage,
								   double horizontalScale, double verticalScale,
								   int blockSizeSrcX, int blockSizeSrcY, int gridSizeSrcX, int gridSizeSrcY, 
								   int blockSizeDstX, int blockSizeDstY, int gridSizeDstX, int gridSizeDstY,
								   void* stream)
{
	//dim3 blockDim(blockSizeSrcX, blockSizeSrcY, 1); // block size = number of threads
 //   dim3 gridDim(gridSizeSrcX, gridSizeSrcY); // grid size = number of blocks

	__wchar_t* msg = 0;

	switch (subPixelType)
	{
	case 1:// 8bit (0...0xff)
		{
		msg = RunFastDownSampleKernel_cpu<unsigned char>(deviceInputBuffer, deviceIntegerOutputBuffer, 
								    deviceOutputBuffer, maxValue, convertValue, numberOfChannels, subPixelType,
								    widthSourceImage, heightSourceImage, strideSourceImage,
								    widthDestImage, heightDestImage, strideIntegerDestImage, strideDestImage,
								    horizontalScale,  verticalScale,
								    blockSizeSrcX, blockSizeSrcY, gridSizeSrcX, gridSizeSrcY, 
								    blockSizeDstX, blockSizeDstY, gridSizeDstX, gridSizeDstY, stream);

			break;
		}
	case 2:// 16bit (0...0xffff)
		{
			msg = RunFastDownSampleKernel_cpu<unsigned short>(deviceInputBuffer, deviceIntegerOutputBuffer, 
						    deviceOutputBuffer, maxValue, convertValue, numberOfChannels, subPixelType,
						    widthSourceImage, heightSourceImage, strideSourceImage,
						    widthDestImage, heightDestImage, strideIntegerDestImage, strideDestImage,
						    horizontalScale,  verticalScale,
						    blockSizeSrcX, blockSizeSrcY, gridSizeSrcX, gridSizeSrcY, 
						    blockSizeDstX, blockSizeDstY, gridSizeDstX, gridSizeDstY, stream);
			break;
		}
	case 4: // Float (0...1)
		{
			msg = RunFastDownSampleKernel_cpu<float>(deviceInputBuffer, deviceIntegerOutputBuffer, 
						    deviceOutputBuffer, maxValue, convertValue, numberOfChannels, subPixelType,
						    widthSourceImage, heightSourceImage, strideSourceImage,
						    widthDestImage, heightDestImage, strideIntegerDestImage, strideDestImage,
						    horizontalScale,  verticalScale,
						    blockSizeSrcX, blockSizeSrcY, gridSizeSrcX, gridSizeSrcY, 
						    blockSizeDstX, blockSizeDstY, gridSizeDstX, gridSizeDstY, stream);
			break;
		}
	default:
		return L"Unsupported sub pixel format";
	}
	return msg;
}