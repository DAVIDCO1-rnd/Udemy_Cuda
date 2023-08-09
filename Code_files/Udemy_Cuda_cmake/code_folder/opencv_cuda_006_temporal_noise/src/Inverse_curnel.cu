#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#pragma once

#include "Utils_device.cu"

//defines a global function called from the host (CPU) excuted on the device (GPU)
template<class T> __global__ void InvertImageKernel(unsigned char* inputData, unsigned char* outputData,
													T white, int alphaChannelNum, int pixelSize, int channelSize,
													int widthImage, int heightImage, 
													int strideSourceImage, int strideResultImage)
{
	int row = 0;
	int column = 0;
	int channel = 0;
	if (!DecodeYXC(&row, &column, &channel, widthImage, heightImage))
		return;	
	
	int indexDst = PixelOffset(row, column, channel , strideResultImage, pixelSize, channelSize); 
	int indexSrc = PixelOffset(row, column, channel , strideSourceImage, pixelSize, channelSize); 
	
	if (channel != alphaChannelNum) // Not alpha channel
	{
		* (Pixel<T>(outputData,indexDst)) = white - * (Pixel<T>(inputData,indexSrc)) ; // Inverse
	}
	else // Alpha Channel
	{
		* (Pixel<T>(outputData,indexDst)) = * (Pixel<T>(inputData,indexSrc)); // Copy 
	}

}
