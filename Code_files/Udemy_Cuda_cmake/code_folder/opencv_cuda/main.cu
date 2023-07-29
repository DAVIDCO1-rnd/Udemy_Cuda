#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

bool read_image_from_file = true;

enum class DirectionOfRotation {
    Clockwise,
    CounterClockwise
};


void print_pixels(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2)
{
    printf("%s\n", matrix_name.c_str());
    for (int i = 0; i < dimension1; i++)
    {
        for (int j = 0; j < dimension2; j++)
        {
            int current_index = i * dimension2 + j;
            unsigned char current_val = pixelData[current_index];
            printf("%d, ", current_val);
        }
        printf("\n");
    }
    printf("\n\n");
}

__global__ void build_image_rotated_by_90_degrees_cuda(unsigned char* device_inputData, unsigned char* device_outputData, int* device_input_width, int* device_input_height)
{
    int input_width = device_input_width[0];
    int input_height = device_input_height[0];
    int output_width = input_height;
    int output_height = input_width;

    int i = threadIdx.x;
    while (i < input_width)
    {
        int j = blockIdx.x;
        while (j < input_height)
        {
            int current_index_input_data = j * input_width + i;
            unsigned char current_val = device_inputData[current_index_input_data];
            int current_index_output_data;
            current_index_output_data = (i + 1) * output_width - j - 1; //clockwise
            //current_index_output_data = (output_height - i - 1) * output_width + j; //counterclockwise
            device_outputData[current_index_output_data] = current_val;
            j += gridDim.x;
        }
        i += blockDim.x;
    }
}

//void build_image_rotated_by_90_degrees_cpu(unsigned char* inputData, unsigned char* outputData, int input_width, int input_height, DirectionOfRotation direction_of_rotation)
//{
//    int output_width = input_height;
//    int output_height = input_width;
//
//    for (int i = 0; i < input_width; i++)
//    {
//        for (int j = 0; j < input_height; j++)
//        {
//            int current_index_input_data = j * input_width + i;
//            unsigned char current_val = inputData[current_index_input_data];
//
//            int current_index_output_data;
//            if (direction_of_rotation == DirectionOfRotation::Clockwise)
//            {
//                current_index_output_data = (i + 1) * output_width - j - 1;
//            }
//            else
//            {
//                current_index_output_data = (output_height - i - 1) * output_width + j;
//            }
//            
//            outputData[current_index_output_data] = current_val;
//            if (read_image_from_file == false)
//            {
//                printf("%d, ", current_index_output_data);
//            }            
//        }
//        if (read_image_from_file == false)
//        {
//            printf("\n");
//        }
//    }
//
//    if (read_image_from_file == false)
//    {
//        printf("\n\n");
//        printf("build_transposed_image_cpu\n");
//        for (int i = 0; i < input_width * input_height; i++)
//        {
//            unsigned char current_val = outputData[i];
//            printf("%d.  %d\n", i, current_val);
//        }
//        printf("\n\n");
//    }  
//}

int main()
{
    //going back from this folder: ./build/code_folder/Section3.3_spotlights/
    std::string image_path = "../../../code_folder/opencv_cuda/images/balloons.jpg";
    cv::Mat image1;
    if (read_image_from_file == true)
    {
        cv::Mat rgb_image1 = cv::imread(image_path);        
        cv::cvtColor(rgb_image1, image1, cv::COLOR_BGR2GRAY);
        if (image1.empty())
        {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 1;
        }
    }


    if (read_image_from_file == false)
    {
        const int width3 = 6;
        const int height3 = 3;

        uchar image_data[height3][width3] = {
           {10, 20, 30, 40, 50, 60},
           {70, 80, 90, 100, 110, 120},
           {130, 140, 150, 160, 170, 180}
        };
        image1 = cv::Mat(height3, width3, CV_8UC1);
        for (int y = 0; y < image1.rows; ++y) {
            for (int x = 0; x < image1.cols; ++x) {
                image1.at<uchar>(y, x) = static_cast<uchar>(image_data[y][x]);
            }
        }

        printf("image1:\n");
        for (int i = 0; i < image1.rows * image1.cols; i++)
        {
            printf("%d.  %d\n", i, image1.data[i]);
        }
        printf("\n\n");
    }



    cv::Mat image2(image1.cols, image1.rows, CV_8UC1);


    unsigned char* pixelData1 = image1.data;
    int height1 = image1.rows;
    int width1 = image1.cols;


    DirectionOfRotation direction_of_rotation = DirectionOfRotation::Clockwise;
    //build_image_rotated_by_90_degrees_cpu(image1.data, image2.data, image1.cols, image1.rows, direction_of_rotation);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create device_inputData
    unsigned char* device_inputData = NULL;
    size_t device_inputData_bytes = sizeof(unsigned char) * image1.rows * image1.cols;
    cudaError_t cudaStatus_inputData_alloc = cudaMalloc((void**)&device_inputData, device_inputData_bytes);
    if (cudaStatus_inputData_alloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return;
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaError_t cudaStatus_g2dim1_memcpy = cudaMemcpy(device_inputData, image1.data, device_inputData_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus_g2dim1_memcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //free_arrays;
        return;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned char* device_outputData = NULL;
    unsigned int device_outputData_num_of_elements = image1.rows * image1.cols;
    size_t device_outputData_num_of_bytes = device_outputData_num_of_elements * sizeof(unsigned char);
    cudaError_t cudaStatus_outputData_alloc = cudaMalloc((void**)&device_outputData, device_outputData_num_of_bytes);
    if (cudaStatus_outputData_alloc != cudaSuccess) {
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
    cudaError_t cudaStatus_input_width_memcpy = cudaMemcpy(device_input_width, &(image1.cols), device_input_width_bytes, cudaMemcpyHostToDevice);
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
    cudaError_t cudaStatus_input_height_memcpy = cudaMemcpy(device_input_height, &(image1.rows), device_input_height_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus_input_height_memcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //free_arrays;
        return;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = 256;

    build_image_rotated_by_90_degrees_cuda << < blocksPerGrid, threadsPerBlock >> > (device_inputData, device_outputData, device_input_width, device_input_height);

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
    unsigned char* outputData = (unsigned char*)malloc(device_outputData_num_of_bytes);
    cudaError_t cudaStatus_outputData_memcpy = cudaMemcpy(outputData, device_outputData, device_outputData_num_of_bytes, cudaMemcpyDeviceToHost);
    if (cudaStatus_outputData_memcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy outputData failed!");
        //free_arrays
        return;
    }
    image2.data = outputData;


    if (read_image_from_file == true)
    {
        cv::imshow("image1", image1);
        cv::imshow("image2", image2);
    }
    else
    {
        print_pixels("image1", image1.data, image1.rows, image1.cols);
        print_pixels("image2", image2.data, image2.rows, image2.cols);
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