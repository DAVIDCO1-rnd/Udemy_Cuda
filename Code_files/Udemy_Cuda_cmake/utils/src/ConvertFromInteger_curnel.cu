#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#pragma once

#include "Utils_device.cu"
// #include "cutil_math.h"

template<class T, int numOfChannels>
__global__ void ConvertFromInteger(unsigned char* inputData,
										  unsigned char* outputData,
										  float saturation, float convertValue,
										  int strideSourceImage,
										  int strideResultImage, int channelSize,
										  int widthImage, int heightImage)
{
	int row = 0;
	int column = 0;
	if (!DecodeYX(&row, &column, widthImage, heightImage))
		return;

	float4 value;
	int index = PixelOffset(row, column, 0 , strideSourceImage, GRAYFLOAT_SIZE * numOfChannels, GRAYFLOAT_SIZE);
	
	value.x = (float)(*(Pixel<int>(inputData, index))) * convertValue;
	if (numOfChannels > 1)
	{
		value.y = (float)(*(Pixel<int>(inputData, index + GRAYFLOAT_SIZE))) * convertValue;
		value.z = (float)(*(Pixel<int>(inputData, index + 2 * GRAYFLOAT_SIZE))) * convertValue;
		if (numOfChannels > 3)
			value.w =(float)(*(Pixel<int>(inputData, index + 3 * GRAYFLOAT_SIZE))) * convertValue;
	}


	__syncthreads();

	index = PixelOffset(row, column, 0 , strideResultImage, channelSize * numOfChannels, channelSize);

	*(Pixel<T>(outputData, index)) =  RoundAndLimitResult<T>(value.x, saturation);
	if (numOfChannels > 1)
	{
		*(Pixel<T>(outputData, index + channelSize)) =  RoundAndLimitResult<T>(value.y, saturation);
		*(Pixel<T>(outputData, index + 2 * channelSize)) =  RoundAndLimitResult<T>(value.z, saturation);
		if (numOfChannels > 3)
			*(Pixel<T>(outputData, index + 3 * channelSize)) = RoundAndLimitResult<T>(value.w, saturation);
	}

}

template<int numOfChannels>
__global__ void ConvertFromIntegerToFloat(unsigned char* data, float convertValue,
										  int strideImage, int widthImage, int heightImage)
{
	int row = 0;
	int column = 0;
	if (!DecodeYX(&row, &column, widthImage, heightImage))
		return;

	float4 value;
	int index = PixelOffset(row, column, 0 , strideImage, GRAYFLOAT_SIZE * numOfChannels, GRAYFLOAT_SIZE);
	value.x = ((float)(*(Pixel< int>(data, index))) * convertValue);
	if (numOfChannels > 1)
	{
		value.y = ((float)(*(Pixel<int>(data, index + GRAYFLOAT_SIZE))) * convertValue);
		value.z = ((float)(*(Pixel<int>(data, index + 2 * GRAYFLOAT_SIZE))) * convertValue);
		if (numOfChannels > 3)
			value.w = ((float)(*(Pixel<int>(data, index + 3 * GRAYFLOAT_SIZE))) * convertValue);
	}


	__syncthreads();

	*(Pixel<float>(data, index)) = value.x;
	if (numOfChannels > 1)
	{
		*(Pixel<float>(data, index + GRAYFLOAT_SIZE)) = value.y;
		*(Pixel<float>(data, index + 2 * GRAYFLOAT_SIZE)) = value.z;
		if (numOfChannels > 3)
			*(Pixel<float>(data, index + 3 * GRAYFLOAT_SIZE)) = value.w;
	}

}

template<int numOfChannels>
__global__ void ConvertFromIntegerToFloat(unsigned char* inputData,
										  unsigned char* outputData,
										  float saturation, float convertValue,
										  int strideSourceImage,
										  int strideResultImage, 
										  int widthImage, int heightImage)
{
	int row = 0;
	int column = 0;
	if (!DecodeYX(&row, &column, widthImage, heightImage))
		return;

	float4 value;
	int index = PixelOffset(row, column, 0 , strideSourceImage, GRAYFLOAT_SIZE * numOfChannels, GRAYFLOAT_SIZE);
	
	value.x = LimitResult<float>
		(((float)(*(Pixel<int>(inputData, index))) * convertValue), saturation);
	if (numOfChannels > 1)
	{
		value.y = LimitResult<float>
			(((float)(*(Pixel<int>(inputData, index + GRAYFLOAT_SIZE))) * convertValue), saturation);
		value.z = LimitResult<float>
			(((float)(*(Pixel<int>(inputData, index + 2 * GRAYFLOAT_SIZE))) * convertValue), saturation);
		if (numOfChannels > 3)
			value.w = LimitResult<float>(((float)(*(Pixel<int>(inputData, index + 3 * GRAYFLOAT_SIZE))) * convertValue), saturation);
	}


	__syncthreads();

	index = PixelOffset(row, column, 0 , strideResultImage, GRAYFLOAT_SIZE * numOfChannels, GRAYFLOAT_SIZE);

	*(Pixel<float>(outputData, index)) = value.x;
	if (numOfChannels > 1)
	{
		*(Pixel<float>(outputData, index + GRAYFLOAT_SIZE)) = value.y;
		*(Pixel<float>(outputData, index + 2 * GRAYFLOAT_SIZE)) = value.z;
		if (numOfChannels > 3)
			*(Pixel<float>(outputData, index + 3 * GRAYFLOAT_SIZE)) = value.w;
	}

}