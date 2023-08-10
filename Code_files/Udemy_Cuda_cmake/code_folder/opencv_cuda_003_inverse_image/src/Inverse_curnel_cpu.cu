#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#pragma once

#include "Utils_device_cpu.cu"

//defines a global function called from the host (CPU) excuted on the device (GPU)
template<class T> void InvertImageKernel_cpu(unsigned char* inputData, unsigned char* outputData,
													T white, int alphaChannelNum, int pixelSize, int channelSize,
													int widthImage, int heightImage, 
													int strideSourceImage, int strideResultImage,
													int blockSizeX, int blockSizeY, int blockSizeZ,
													int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_x; thread_Idx_x < blockSizeX; thread_Idx_x++)
	{
		for (int thread_Idx_y; thread_Idx_y < blockSizeY; thread_Idx_y++)
		{
			for (int thread_Idx_z; thread_Idx_z < blockSizeZ; thread_Idx_z++)
			{
				for (int block_Idx_x; block_Idx_x < gridSizeX; block_Idx_x++)
				{
					for (int block_Idx_y; block_Idx_y < gridSizeY; block_Idx_y++)
					{
						int row = 0;
						int column = 0;
						int channel = 0;
						if (!DecodeYXC_cpu(&row, &column, &channel, widthImage, heightImage, thread_Idx_x, thread_Idx_y, thread_Idx_z, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
							return;

						int indexDst = PixelOffset_cpu(row, column, channel, strideResultImage, pixelSize, channelSize);
						int indexSrc = PixelOffset_cpu(row, column, channel, strideSourceImage, pixelSize, channelSize);

						if (channel != alphaChannelNum) // Not alpha channel
						{
							*(Pixel_cpu<T>(outputData, indexDst)) = white - *(Pixel_cpu<T>(inputData, indexSrc)); // Inverse
						}
						else // Alpha Channel
						{
							*(Pixel_cpu<T>(outputData, indexDst)) = *(Pixel_cpu<T>(inputData, indexSrc)); // Copy 
						}
					}
				}
			}
		}
	}
}
