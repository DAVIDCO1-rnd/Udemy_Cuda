#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#pragma once

#include "Utils_device.cu"

//#define USE_X_DIMENSIONS_ONLY


template<class T> __global__ void rotate_90_kernel(unsigned char* device_inputData, unsigned char* device_outputData, 
    int input_width, int input_height, int pixel_size, int is_clockwise)
{
    int output_width = input_height;
    int output_height = input_width;

#ifdef USE_X_DIMENSIONS_ONLY
    int x = threadIdx.x;
    while (x < input_width)
    {
        int y = blockIdx.x;
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
            T pixel_value = *(T*)(device_inputData + current_index_output_data);
            *((T*)(device_outputData + current_index_input_data)) = pixel_value;
            y += gridDim.x;
        }
        x += blockDim.x;
    }
#else //USE_X_DIMENSIONS_ONLY
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
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
        T pixel_value = *(T*)(device_inputData + current_index_output_data);
        *((T*)(device_outputData + current_index_input_data)) = pixel_value;
    }
#endif //USE_X_DIMENSIONS_ONLY
}

