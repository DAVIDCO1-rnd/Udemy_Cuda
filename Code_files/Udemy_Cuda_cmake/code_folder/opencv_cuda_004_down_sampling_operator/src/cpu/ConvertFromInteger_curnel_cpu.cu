#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#pragma once

#include "Utils_device_cpu.cu"
#include "fake_cuda_in_cpu.cuh"

template<class T, int numOfChannels>
void ConvertFromInteger_cpu(unsigned char* inputData,
										  unsigned char* outputData,
										  float saturation, float convertValue,
										  int strideSourceImage,
										  int strideResultImage, int channelSize,
										  int widthImage, int heightImage,
											int blockSizeX, int blockSizeY,
											int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
			{
				for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
				{
					int row = 0;
					int column = 0;
					if (!DecodeYX_cpu(&row, &column, widthImage, heightImage, thread_Idx_x, thread_Idx_y, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
						break;

					float4 value;
					int index = PixelOffset_cpu(row, column, 0, strideSourceImage, GRAYFLOAT_SIZE * numOfChannels, GRAYFLOAT_SIZE);

					value.x = (float)(*(Pixel_cpu<int>(inputData, index))) * convertValue;
					if (numOfChannels > 1)
					{
						value.y = (float)(*(Pixel_cpu<int>(inputData, index + GRAYFLOAT_SIZE))) * convertValue;
						value.z = (float)(*(Pixel_cpu<int>(inputData, index + 2 * GRAYFLOAT_SIZE))) * convertValue;
						if (numOfChannels > 3)
							value.w = (float)(*(Pixel_cpu<int>(inputData, index + 3 * GRAYFLOAT_SIZE))) * convertValue;
					}


					syncthreads_cpu();

					index = PixelOffset_cpu(row, column, 0, strideResultImage, channelSize * numOfChannels, channelSize);

					*(Pixel_cpu<T>(outputData, index)) = RoundAndLimitResult_cpu<T>(value.x, saturation);
					if (numOfChannels > 1)
					{
						*(Pixel_cpu<T>(outputData, index + channelSize)) = RoundAndLimitResult_cpu<T>(value.y, saturation);
						*(Pixel_cpu<T>(outputData, index + 2 * channelSize)) = RoundAndLimitResult_cpu<T>(value.z, saturation);
						if (numOfChannels > 3)
							*(Pixel_cpu<T>(outputData, index + 3 * channelSize)) = RoundAndLimitResult_cpu<T>(value.w, saturation);
					}
				}
			}
		}
	}
}

template<int numOfChannels>
void ConvertFromIntegerToFloat_cpu(unsigned char* data, float convertValue,
									int strideImage, int widthImage, int heightImage,
									int blockSizeX, int blockSizeY,
									int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
			{
				for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
				{
					int row = 0;
					int column = 0;
					if (!DecodeYX_cpu(&row, &column, widthImage, heightImage, thread_Idx_x, thread_Idx_y, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
						break;

					float4 value;
					int index = PixelOffset_cpu(row, column, 0, strideImage, GRAYFLOAT_SIZE * numOfChannels, GRAYFLOAT_SIZE);
					value.x = ((float)(*(Pixel_cpu< int>(data, index))) * convertValue);
					if (numOfChannels > 1)
					{
						value.y = ((float)(*(Pixel_cpu<int>(data, index + GRAYFLOAT_SIZE))) * convertValue);
						value.z = ((float)(*(Pixel_cpu<int>(data, index + 2 * GRAYFLOAT_SIZE))) * convertValue);
						if (numOfChannels > 3)
							value.w = ((float)(*(Pixel_cpu<int>(data, index + 3 * GRAYFLOAT_SIZE))) * convertValue);
					}


					syncthreads_cpu();

					*(Pixel_cpu<float>(data, index)) = value.x;
					if (numOfChannels > 1)
					{
						*(Pixel_cpu<float>(data, index + GRAYFLOAT_SIZE)) = value.y;
						*(Pixel_cpu<float>(data, index + 2 * GRAYFLOAT_SIZE)) = value.z;
						if (numOfChannels > 3)
							*(Pixel_cpu<float>(data, index + 3 * GRAYFLOAT_SIZE)) = value.w;
					}
				}
			}
		}
	}
}

template<int numOfChannels>
void ConvertFromIntegerToFloat_cpu(unsigned char* inputData,
										  unsigned char* outputData,
										  float saturation, float convertValue,
										  int strideSourceImage,
										  int strideResultImage, 
										  int widthImage, int heightImage,
											int blockSizeX, int blockSizeY,
											int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
			{
				for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
				{
					int row = 0;
					int column = 0;
					if (!DecodeYX_cpu(&row, &column, widthImage, heightImage, thread_Idx_x, thread_Idx_y, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
						break;

					float4 value;
					int index = PixelOffset_cpu(row, column, 0, strideSourceImage, GRAYFLOAT_SIZE * numOfChannels, GRAYFLOAT_SIZE);

					value.x = LimitResult_cpu<float>
						(((float)(*(Pixel_cpu<int>(inputData, index))) * convertValue), saturation);
					if (numOfChannels > 1)
					{
						value.y = LimitResult_cpu<float>
							(((float)(*(Pixel_cpu<int>(inputData, index + GRAYFLOAT_SIZE))) * convertValue), saturation);
						value.z = LimitResult_cpu<float>
							(((float)(*(Pixel_cpu<int>(inputData, index + 2 * GRAYFLOAT_SIZE))) * convertValue), saturation);
						if (numOfChannels > 3)
							value.w = LimitResult_cpu<float>(((float)(*(Pixel_cpu<int>(inputData, index + 3 * GRAYFLOAT_SIZE))) * convertValue), saturation);
					}


					syncthreads_cpu();

					index = PixelOffset_cpu(row, column, 0, strideResultImage, GRAYFLOAT_SIZE * numOfChannels, GRAYFLOAT_SIZE);

					*(Pixel_cpu<float>(outputData, index)) = value.x;
					if (numOfChannels > 1)
					{
						*(Pixel_cpu<float>(outputData, index + GRAYFLOAT_SIZE)) = value.y;
						*(Pixel_cpu<float>(outputData, index + 2 * GRAYFLOAT_SIZE)) = value.z;
						if (numOfChannels > 3)
							*(Pixel_cpu<float>(outputData, index + 3 * GRAYFLOAT_SIZE)) = value.w;
					}
				}
			}
		}
	}
}