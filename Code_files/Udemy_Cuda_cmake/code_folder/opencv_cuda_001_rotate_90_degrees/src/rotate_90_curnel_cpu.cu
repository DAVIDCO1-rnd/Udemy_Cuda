#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#pragma once

#include "Utils_device_cpu.cu"

//#define USE_X_DIMENSIONS_ONLY

//defines a global function called from the host (CPU) excuted on the device (GPU)
template<class T> void rotate_90_kernel_cpu(unsigned char* inputData, unsigned char* outputData,
													int input_width, int input_height, int pixel_size, int is_clockwise,
													int blockSizeX, int blockSizeY, int blockSizeZ,
													int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int thread_Idx_z = 0; thread_Idx_z < blockSizeZ; thread_Idx_z++)
			{
				for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
				{
					for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
					{


                        int output_width = input_height;
                        int output_height = input_width;

#ifdef USE_X_DIMENSIONS_ONLY
                        int x = thread_Idx_x;
                        while (x < input_width)
                        {
                            int y = block_Idx_x;
                            while (y < input_height)
                            {
                                int current_index_input_data = pixel_size * (x * input_height + y);
                                int current_index_output_data;
                                if (is_clockwise == 1)
                                {
                                    current_index_output_data = pixel_size * ((input_height - y - 1) * input_width + x); //Clockwise
                                }
                                else
                                {
                                    current_index_output_data = pixel_size * (y * input_width + input_width - 1 - x); //CounterClockwise
                                }
                                T pixel_value = *(T*)(inputData + current_index_output_data);
                                *((T*)(outputData + current_index_input_data)) = pixel_value;
                                y += gridSizeX;
                            }
                            x += blockSizeX;
                        }
#else //USE_X_DIMENSIONS_ONLY
                        int x = thread_Idx_x + block_Idx_x * block_Dim_x;
                        int y = thread_Idx_y + block_Idx_y * block_Dim_y;
                        int current_index_input_data = pixel_size * (y + x * input_height);
                        if (x < input_width && y < input_height)
                        {
                            int current_index_output_data;
                            if (is_clockwise == 1)
                            {
                                current_index_output_data = pixel_size * ((input_height - y - 1) * input_width + x); //Clockwise
                            }
                            else
                            {
                                current_index_output_data = pixel_size * (y * input_width + input_width - 1 - x); //CounterClockwise
                            }
                            T pixel_value = *(T*)(inputData + current_index_output_data);
                            *((T*)(outputData + current_index_input_data)) = pixel_value;
                        }
#endif //USE_X_DIMENSIONS_ONLY

						//int row = 0;
						//int column = 0;
						//int channel = 0;
						//if (!DecodeYXC_cpu(&row, &column, &channel, widthImage, heightImage, thread_Idx_x, thread_Idx_y, thread_Idx_z, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
						//	return;

						//int indexDst = PixelOffset_cpu(row, column, channel, strideResultImage, pixelSize, channelSize);
						//int indexSrc = PixelOffset_cpu(row, column, channel, strideSourceImage, pixelSize, channelSize);

						//T current_val = *(Pixel_cpu<T>(inputData, indexSrc));
						//if (channel != alphaChannelNum) // Not alpha channel
						//{	
						//	T new_val = white - current_val;
						//	*(Pixel_cpu<T>(outputData, indexDst)) = new_val; // Inverse
						//}
						//else // Alpha Channel
						//{
						//	*(Pixel_cpu<T>(outputData, indexDst)) = current_val; // Copy 
						//}
					}
				}
			}
		}
	}
}