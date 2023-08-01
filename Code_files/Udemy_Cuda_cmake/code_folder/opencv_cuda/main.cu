#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <stdio.h>

#define USE_CUDA

#ifdef USE_CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif //USE_CUDA

bool read_image_from_file = true;
const int height = 3;
const int width = 5;

enum class DirectionOfRotation {
    Clockwise = 0,
    CounterClockwise = 1
};

enum class PixelType {
    UCHAR = 1,
    USHORT = 2,
    FLOAT = 4
};

void print_single_val(unsigned char* pixelData, int i, PixelType pixel_type)
{
    if (pixel_type == PixelType::UCHAR)
    {
        unsigned char current_val = pixelData[i];
        printf("0x%02x, ", current_val);
    }
    if (pixel_type == PixelType::USHORT)
    {
        unsigned char sub_pixel1 = pixelData[i + 0];
        unsigned char sub_pixel2 = pixelData[i + 1];
        unsigned short current_val = 0x100 * sub_pixel2 + sub_pixel1;
        printf("0x%04x, ", current_val);
    }
    if (pixel_type == PixelType::FLOAT)
    {
        unsigned char sub_pixel1 = pixelData[i + 0];
        unsigned char sub_pixel2 = pixelData[i + 1];
        unsigned char sub_pixel3 = pixelData[i + 2];
        unsigned char sub_pixel4 = pixelData[i + 3];
        float current_val = 4.0 * sub_pixel4 + 3.0 * sub_pixel3 + 2.0 * sub_pixel2 + 1.0 * sub_pixel1;
        printf("%f, ", current_val);
    }
}


void print_pixels_1D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type)
{
    int pixel_size = (int)pixel_type;
    printf("%s as 1D array\n", matrix_name.c_str());
    for (int i = 0; i < pixel_size * dimension1 * dimension2; i+=pixel_size)
    {
        print_single_val(pixelData, i, pixel_type);
    }
    printf("\n\n");
}

void print_pixels_2D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type)
{
    int pixel_size = (int)pixel_type;
    printf("%s as 2D array\n", matrix_name.c_str());
    for (int i = 0; i < dimension1; i++)
    {
        for (int j = 0; j < pixel_size * dimension2; j += pixel_size)
        {
            int current_index = i * pixel_size * dimension2 + j;
            print_single_val(pixelData, current_index, pixel_type);
        }
        printf("\n");
    }
    printf("\n\n");
}

void print_pixels(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type)
{
    print_pixels_1D(matrix_name, pixelData, dimension1, dimension2, pixel_type);
    print_pixels_2D(matrix_name, pixelData, dimension1, dimension2, pixel_type);
}

#ifdef USE_CUDA
template<class T> __global__ void build_image_rotated_by_90_degrees_cuda(unsigned char* device_inputData, unsigned char* device_outputData, int* device_input_width, int* device_input_height, int* device_pixel_size)
{
    int input_width = device_input_width[0];
    int input_height = device_input_height[0];
    int output_width = input_height;
    int output_height = input_width;
    int pixel_size = device_pixel_size[0];

    int i = threadIdx.x;
    while (i < input_width)
    {
        int j = blockIdx.x;
        while (j < input_height)
        {
            int current_index_input_data = pixel_size * (i * input_height + j);
            int current_index_output_data = pixel_size * ((input_height - j - 1) * input_width + i); //Clockwise
            //int current_index_output_data = pixel_size * (j * input_width + input_width - 1 - i); //CounterClockwise
            *((T*)(device_outputData + current_index_input_data)) = *(T*)(device_inputData + current_index_output_data);
            j += gridDim.x;
        }
        i += blockDim.x;
    }
}
#endif //USE_CUDA

template <typename T>
void build_image_rotated_by_90_degrees_cpu(unsigned char* inputData, unsigned char* outputData, int input_width, int input_height, int pixel_size, int direction_of_rotation)
{
    int output_width = input_height;
    int output_height = input_width;

    for (int i = 0; i < input_width; i++)
    {
        for (int j = 0; j < input_height; j++)
        {
            int current_index_input_data = pixel_size * (i * input_height + j);
            int current_index_output_data;

            if (direction_of_rotation == 0) //Clockwise
            {
                current_index_output_data = pixel_size * ((input_height - j - 1) * input_width + i);
            }
            else //CounterClockwise
            {
                current_index_output_data = pixel_size * (j * input_width + input_width - 1 - i);
            }
            *((T*)(outputData + current_index_input_data)) = *(T*)(inputData + current_index_output_data);
            
            if (read_image_from_file == false)
            {
                printf("%d, ", current_index_output_data);
            }            
        }
        if (read_image_from_file == false)
        {
            printf("\n");
        }
    }

    if (read_image_from_file == false)
    {
        printf("\n\n");
        printf("build_transposed_image_cpu\n");
        for (int i = 0; i < input_width * input_height; i++)
        {
            unsigned char current_val = outputData[i];
            printf("%d.  %d\n", i, current_val);
        }
        printf("\n\n");
    }  
}

cv::Mat build_image_from_data(uchar image_data[][width], PixelType pixel_type)
{
    cv::Mat image;
    switch (pixel_type)
    {
        case PixelType::UCHAR:
            image = cv::Mat(height, width, CV_8UC1);
            for (int y = 0; y < image.rows; ++y) {
                for (int x = 0; x < image.cols; ++x) {
                    image.at<uchar>(y, x) = static_cast<uchar>(image_data[y][x]);
                }
            }
            break;

        case PixelType::USHORT:
            image = cv::Mat(height, width, CV_16UC1);
            for (int y = 0; y < image.rows; ++y) {
                for (int x = 0; x < image.cols; ++x) {
                    uchar current_val = image_data[y][x];
                    ushort current_val_ushort = (ushort)current_val;
                    ushort new_val = 0xFF00 + current_val_ushort;
                    image.at<ushort>(y, x) = new_val;
                }
            }
            break;

        case PixelType::FLOAT:
            image = cv::Mat(height, width, CV_32FC1);
            for (int y = 0; y < image.rows; ++y) {
                for (int x = 0; x < image.cols; ++x) {
                    uchar current_val = image_data[y][x];
                    float current_val_float = (float)current_val;
                    float new_val = 1.0 * current_val_float;
                    image.at<float>(y, x) = new_val;
                }
            }
            break;

    }
    return image;
}

int main()
{
    //going back from this folder: ./build/code_folder/Section3.3_spotlights/
    std::string image_path = "../../../code_folder/opencv_cuda/images/balloons.jpg";
    cv::Mat image1_uchar;
    cv::Mat image1_ushort;
    cv::Mat image1_float;
    if (read_image_from_file == true)
    {
        cv::Mat rgb_image1 = cv::imread(image_path);        
        cv::cvtColor(rgb_image1, image1_uchar, cv::COLOR_BGR2GRAY);
        if (image1_uchar.empty())
        {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 1;
        }
        image1_uchar.convertTo(image1_ushort, CV_16UC1, 256);
        image1_uchar.convertTo(image1_float, CV_32FC1, 65536);
    }


    if (read_image_from_file == false)
    {
        //uchar image_data[height][width] = {
        //   {0x05, 0x10, 0x15, 0x20, 0x25, 0x30},
        //   {0x35, 0x40, 0x45, 0x50, 0x55, 0x60},
        //   {0x65, 0x70, 0x75, 0x80, 0x85, 0x90}
        //};

        uchar image_data[height][width] = {
           {0x00, 0x01, 0x02, 0x03, 0x04},
           {0x05, 0x06, 0x07, 0x08, 0x09},
           {0x10, 0x11, 0x12, 0x13, 0x14}
        };
        image1_uchar = build_image_from_data(image_data, PixelType::UCHAR);        
        print_pixels("built-in image1_uchar", image1_uchar.data, image1_uchar.rows, image1_uchar.cols, PixelType::UCHAR);

        image1_ushort = build_image_from_data(image_data, PixelType::USHORT);
        print_pixels("built-in image1_ushort", image1_ushort.data, image1_ushort.rows, image1_ushort.cols, PixelType::USHORT);

        image1_float = build_image_from_data(image_data, PixelType::FLOAT);
        print_pixels("built-in image1_float", image1_float.data, image1_float.rows, image1_float.cols, PixelType::FLOAT);
    }



    cv::Mat image2_uchar(image1_uchar.cols, image1_uchar.rows, CV_8UC1);
    cv::Mat image2_ushort(image1_ushort.cols, image1_ushort.rows, CV_16UC1);
    cv::Mat image2_float(image1_ushort.cols, image1_ushort.rows, CV_32FC1);



    DirectionOfRotation direction_of_rotation = DirectionOfRotation::Clockwise;
#ifndef USE_CUDA
    build_image_rotated_by_90_degrees_cpu<unsigned char>(image1_uchar.data, image2_uchar.data, image1_uchar.cols, image1_uchar.rows, (int)PixelType::UCHAR, (int)direction_of_rotation);

    build_image_rotated_by_90_degrees_cpu<unsigned short>(image1_ushort.data, image2_ushort.data, image1_ushort.cols, image1_ushort.rows, (int)PixelType::USHORT, (int)direction_of_rotation);

    build_image_rotated_by_90_degrees_cpu<float>(image1_float.data, image2_float.data, image1_float.cols, image1_float.rows, (int)PixelType::FLOAT, (int)direction_of_rotation);
#endif

#ifdef USE_CUDA
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create device_inputData1
    unsigned char* device_inputData1 = NULL;
    size_t device_inputData_bytes1 = sizeof(unsigned char) * image1_uchar.rows * image1_uchar.cols;
    cudaError_t cudaStatus_inputData_alloc1 = cudaMalloc((void**)&device_inputData1, device_inputData_bytes1);
    if (cudaStatus_inputData_alloc1 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return;
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaError_t cudaStatus_inputData_memcpy1 = cudaMemcpy(device_inputData1, image1_uchar.data, device_inputData_bytes1, cudaMemcpyHostToDevice);
    if (cudaStatus_inputData_memcpy1 != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //free_arrays;
        return;
    }

    //create device_inputData2
    unsigned char* device_inputData2 = NULL;
    size_t device_inputData_bytes2 = sizeof(unsigned short) * image1_ushort.rows * image1_ushort.cols;
    cudaError_t cudaStatus_inputData_alloc2 = cudaMalloc((void**)&device_inputData2, device_inputData_bytes2);
    if (cudaStatus_inputData_alloc2 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return;
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaError_t cudaStatus_inputData_memcpy2 = cudaMemcpy(device_inputData2, image1_ushort.data, device_inputData_bytes2, cudaMemcpyHostToDevice);
    if (cudaStatus_inputData_memcpy2 != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //free_arrays;
        return;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned char* device_outputData1 = NULL;
    unsigned char* device_outputData2 = NULL;
    unsigned int device_outputData_num_of_elements = image1_uchar.rows * image1_uchar.cols;
    size_t device_outputData_num_of_bytes1 = device_outputData_num_of_elements * sizeof(unsigned char);
    cudaError_t cudaStatus_outputData_alloc1 = cudaMalloc((void**)&device_outputData1, device_outputData_num_of_bytes1);
    if (cudaStatus_outputData_alloc1 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //free_arrays;
        return;
    }
    size_t device_outputData_num_of_bytes2 = device_outputData_num_of_elements * sizeof(unsigned short);
    cudaError_t cudaStatus_outputData_alloc2 = cudaMalloc((void**)&device_outputData2, device_outputData_num_of_bytes2);
    if (cudaStatus_outputData_alloc2 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //free_arrays;
        return;
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create device_input_width
    int* device_input_width = NULL;
    size_t device_input_width_bytes = sizeof(int);
    cudaError_t cudaStatus_input_width_alloc = cudaMalloc((void**)&device_input_width, device_input_width_bytes);
    if (cudaStatus_input_width_alloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return;
    }
    cudaError_t cudaStatus_input_width_memcpy = cudaMemcpy(device_input_width, &(image1_uchar.cols), device_input_width_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus_input_width_memcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //free_arrays;
        return;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create device_input_height
    int* device_input_height = NULL;
    size_t device_input_height_bytes = sizeof(int);
    cudaError_t cudaStatus_input_height_alloc = cudaMalloc((void**)&device_input_height, device_input_height_bytes);
    if (cudaStatus_input_height_alloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return;
    }
    cudaError_t cudaStatus_input_height_memcpy = cudaMemcpy(device_input_height, &(image1_uchar.rows), device_input_height_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus_input_height_memcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //free_arrays;
        return;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create device_uchar_pixel_size
    int* device_uchar_pixel_size = NULL;
    size_t device_uchar_pixel_size_bytes = sizeof(int);
    cudaError_t cudaStatus_uchar_pixel_size_alloc = cudaMalloc((void**)&device_uchar_pixel_size, device_uchar_pixel_size_bytes);
    if (cudaStatus_uchar_pixel_size_alloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return;
    }
    int uchar_pixel_size = (int)PixelType::UCHAR;
    cudaError_t cudaStatus_uchar_pixel_size_memcpy = cudaMemcpy(device_uchar_pixel_size, &(uchar_pixel_size), device_uchar_pixel_size_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus_uchar_pixel_size_memcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //free_arrays;
        return;
    }

    //create device_ushort_pixel_size
    int* device_ushort_pixel_size = NULL;
    size_t device_ushort_pixel_size_bytes = sizeof(int);
    cudaError_t cudaStatus_ushort_pixel_size_alloc = cudaMalloc((void**)&device_ushort_pixel_size, device_ushort_pixel_size_bytes);
    if (cudaStatus_ushort_pixel_size_alloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return;
    }
    int ushort_pixel_size = (int)PixelType::USHORT;
    cudaError_t cudaStatus_ushort_pixel_size_memcpy = cudaMemcpy(device_ushort_pixel_size, &(ushort_pixel_size), device_ushort_pixel_size_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus_ushort_pixel_size_memcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //free_arrays;
        return;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = 256;

    build_image_rotated_by_90_degrees_cuda<unsigned char> << < blocksPerGrid, threadsPerBlock >> > (device_inputData1, device_outputData1, device_input_width, device_input_height, device_uchar_pixel_size);
    build_image_rotated_by_90_degrees_cuda<unsigned short> << < blocksPerGrid, threadsPerBlock >> > (device_inputData2, device_outputData2, device_input_width, device_input_height, device_ushort_pixel_size);

    // Check for any errors launching the kernel
    cudaError_t cudaStatusLastError = cudaGetLastError();
    if (cudaStatusLastError != cudaSuccess) {
        fprintf(stderr, "build_image_rotated_by_90_degrees_cuda launch failed: %s\n", cudaGetErrorString(cudaStatusLastError));
        //free_arrays
        return;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaError_t cudaStatusDeviceSynchronize = cudaDeviceSynchronize();
    if (cudaStatusDeviceSynchronize != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching build_image_rotated_by_90_degrees_cuda!\n", cudaStatusDeviceSynchronize);
        //free_arrays
        return;
    }

    // Copy output vector from GPU buffer to host memory.
    unsigned char* outputData1 = (unsigned char*)malloc(device_outputData_num_of_bytes1);
    cudaError_t cudaStatus_outputData_memcpy1 = cudaMemcpy(outputData1, device_outputData1, device_outputData_num_of_bytes1, cudaMemcpyDeviceToHost);
    if (cudaStatus_outputData_memcpy1 != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy outputData1 failed!");
        //free_arrays
        return;
    }
    image2_uchar.data = outputData1;

    // Copy output vector from GPU buffer to host memory.
    unsigned char* outputData2 = (unsigned char*)malloc(device_outputData_num_of_bytes2);
    cudaError_t cudaStatus_outputData_memcpy2 = cudaMemcpy(outputData2, device_outputData2, device_outputData_num_of_bytes2, cudaMemcpyDeviceToHost);
    if (cudaStatus_outputData_memcpy2 != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy outputData2 failed!");
        //free_arrays
        return;
    }
    image2_ushort.data = outputData1;
#endif //USE_CUDA


    if (read_image_from_file == true)
    {
        cv::imshow("image1_uchar", image1_uchar);
        cv::imshow("image2_uchar", image2_uchar);

        cv::imshow("image1_ushort", image1_ushort);
        cv::imshow("image2_ushort", image2_ushort);

        //cv::imshow("image1_float", image1_float);
        //cv::imshow("image2_float", image2_float);
    }
    else
    {
        print_pixels("image1_uchar", image1_uchar.data, image1_uchar.rows, image1_uchar.cols, PixelType::UCHAR);
        print_pixels("image2_uchar", image2_uchar.data, image2_uchar.rows, image2_uchar.cols, PixelType::UCHAR);

        print_pixels("image1_ushort", image1_ushort.data, image1_ushort.rows, image1_ushort.cols, PixelType::USHORT);
        print_pixels("image2_ushort", image2_ushort.data, image2_ushort.rows, image2_ushort.cols, PixelType::USHORT);

        //print_pixels("image1_float", image1_ushort.data, image1_ushort.rows, image1_ushort.cols, PixelType::FLOAT);
        //print_pixels("image2_float", image2_ushort.data, image2_ushort.rows, image2_ushort.cols, PixelType::FLOAT);
    }

    int k = cv::waitKey(0); // Wait for a keystroke in the window



    return 0;
}




//#include <iostream>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <cuda_runtime.h>
//
//// CUDA kernel code
//__global__ void multiply_by_constant(float* input, float constant, int size)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < size)
//    {
//        input[idx] *= constant;
//    }
//}
//
//int main()
//{
//    // Create a sample buffer array in C++
//    int size = 9;
//    float input_buffer[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
//
//    // Allocate memory on the GPU
//    float* d_input_buffer;
//    cudaMalloc((void**)&d_input_buffer, size * sizeof(float));
//
//    // Copy the input buffer from the CPU to the GPU
//    cudaMemcpy(d_input_buffer, input_buffer, size * sizeof(float), cudaMemcpyHostToDevice);
//
//    // Define the block and grid dimensions for CUDA execution
//    int block_size = 256;
//    int num_blocks = (size + block_size - 1) / block_size;
//
//    // Execute the CUDA kernel
//    multiply_by_constant<<<num_blocks, block_size>>>(d_input_buffer, 2.0f, size);
//
//    // Copy the result back from the GPU to the CPU
//    float output_buffer[size];
//    cudaMemcpy(output_buffer, d_input_buffer, size * sizeof(float), cudaMemcpyDeviceToHost);
//
//    // Clean up memory on the GPU
//    cudaFree(d_input_buffer);
//
//    // Show the result using OpenCV (just as an example)
//    cv::Mat result = cv::Mat(1, size, CV_32F, output_buffer);
//    std::cout << "Input Buffer: " << cv::Mat(1, size, CV_32F, input_buffer) << std::endl;
//    std::cout << "Output Buffer: " << result << std::endl;
//
//    return 0;
//}