#include <string>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include "rotate_90.cuh"
#include "rotate_90_cpu.cuh"
#include "utils_custom_matrices.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_utils.cuh"
#include "opencv_utils.h"


//#define USE_CUDA
//#define USE_X_DIMENSIONS_ONLY

bool read_image_from_file = true;


#ifndef USE_X_DIMENSIONS_ONLY
enum class ThreadsAndBlocksCalculations {
    Use_optimal_function = 0,
    Use_threads_as_warp_size = 1
};
#endif //USE_X_DIMENSIONS_ONLY


int main()
{
#ifndef USE_X_DIMENSIONS_ONLY
    ThreadsAndBlocksCalculations threads_and_blocks_calculations = ThreadsAndBlocksCalculations::Use_optimal_function;
#endif //USE_X_DIMENSIONS_ONLY


    //going back from this folder: ./build/code_folder/Section3.3_spotlights/
    std::string image_path = "../../../images/balloons_rgb_width_2048_height_2560.jpg";
    cv::Mat image1_uchar;
    cv::Mat image1_ushort;
    cv::Mat image1_float;
    if (read_image_from_file == true)
    {
        cv::Mat rgb_image1 = cv::imread(image_path);
        cv::cvtColor(rgb_image1, image1_uchar, cv::COLOR_BGR2GRAY);
        //cv::vconcat(image1_uchar, image1_uchar, image1_uchar);
        //cv::hconcat(image1_uchar, image1_uchar, image1_uchar);
        //int newWidth = 2048;
        //int newHeight = 2560;
        //cv::resize(rgb_image1, rgb_image1, cv::Size(newWidth, newHeight), cv::INTER_LINEAR);
        //cv::imwrite(image_path, rgb_image1);
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
        image1_uchar = build_image_from_data(PixelType::UCHAR, width, height);
        print_pixels("built-in image1_uchar", image1_uchar.data, image1_uchar.rows, image1_uchar.cols, PixelType::UCHAR, false);

        image1_ushort = build_image_from_data(PixelType::USHORT, width, height);
        print_pixels("built-in image1_ushort", image1_ushort.data, image1_ushort.rows, image1_ushort.cols, PixelType::USHORT, false);

        image1_float = build_image_from_data(PixelType::FLOAT, width, height);
        print_pixels("built-in image1_float", image1_float.data, image1_float.rows, image1_float.cols, PixelType::FLOAT, false);
    }



    cv::Mat image2_uchar(image1_uchar.cols, image1_uchar.rows, CV_8UC1);
    cv::Mat image2_ushort(image1_ushort.cols, image1_ushort.rows, CV_16UC1);
    cv::Mat image2_float(image1_float.cols, image1_float.rows, CV_32FC1);


    unsigned char max_val_uchar = 255;
    int alphaChannelNum = -1;
    int uchar_channelSize = 1;
    int input_image_width = image1_uchar.cols;
    int input_image_height = image1_uchar.rows;
    int uchar_pixel_size = (int)PixelType::UCHAR;
    int uchar_strideSourceImage = input_image_width * uchar_pixel_size;
    int uchar_strideResultImage = input_image_width * uchar_pixel_size;
    int uchar_subPixelType = 1;


    unsigned short max_val_ushort = 65535;
    int ushort_subPixelType = 2;
    int ushort_channelSize = 2;    
    int ushort_pixel_size = (int)PixelType::USHORT;
    int ushort_strideSourceImage = input_image_width * ushort_pixel_size;
    int ushort_strideResultImage = input_image_width * ushort_pixel_size;

    int image_height = image1_uchar.rows;
    int image_width = image1_uchar.cols;
    int num_of_channels = 1;
    int is_clockwise = 1;
    BlockAndGridDimensions* block_and_grid_dims = CalculateBlockAndGridDimensions(num_of_channels, image_width, image_height);

    dim3 blocksPerGrid;
    dim3 threadsPerBlock;

    auto start_time_cpu = std::chrono::high_resolution_clock::now();
#ifdef USE_CUDA
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create device_inputData1
    unsigned char* device_inputData1 = NULL;
    size_t device_inputData_bytes1 = sizeof(unsigned char) * image1_uchar.rows * image1_uchar.cols;
    HANDLE_ERROR(cudaMalloc((void**)&device_inputData1, device_inputData_bytes1));
    HANDLE_ERROR(cudaMemcpy(device_inputData1, image1_uchar.data, device_inputData_bytes1, cudaMemcpyHostToDevice));


    //create device_inputData2
    unsigned char* device_inputData2 = NULL;
    size_t device_inputData_bytes2 = sizeof(unsigned short) * image1_ushort.rows * image1_ushort.cols;
    HANDLE_ERROR(cudaMalloc((void**)&device_inputData2, device_inputData_bytes2));

    // Copy input vectors from host memory to GPU buffers.
    HANDLE_ERROR(cudaMemcpy(device_inputData2, image1_ushort.data, device_inputData_bytes2, cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned char* device_outputData1 = NULL;
    unsigned char* device_outputData2 = NULL;
    unsigned int device_outputData_num_of_elements = image1_uchar.rows * image1_uchar.cols;
    size_t device_outputData_num_of_bytes1 = device_outputData_num_of_elements * sizeof(unsigned char);
    HANDLE_ERROR(cudaMalloc((void**)&device_outputData1, device_outputData_num_of_bytes1));

    size_t device_outputData_num_of_bytes2 = device_outputData_num_of_elements * sizeof(unsigned short);
    HANDLE_ERROR(cudaMalloc((void**)&device_outputData2, device_outputData_num_of_bytes2));

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_X_DIMENSIONS_ONLY
    blocksPerGrid = dim3(256, 1, 1);
    threadsPerBlock = dim3(256, 1, 1);
#else //USE_X_DIMENSIONS_ONLY
    if (threads_and_blocks_calculations == ThreadsAndBlocksCalculations::Use_optimal_function)
    {        
        blocksPerGrid = block_and_grid_dims->blocksPerGrid;
        threadsPerBlock = block_and_grid_dims->threadsPerBlock;
    }
    else if (threads_and_blocks_calculations == ThreadsAndBlocksCalculations::Use_threads_as_warp_size)
    {
        int num_of_threads_x = 32;
        int num_of_threads_y = 32;
        int num_of_blocks_x = (image_width + num_of_threads_x - 1) / num_of_threads_x;
        int num_of_blocks_y = (image_height + num_of_threads_y - 1) / num_of_threads_y;
        blocksPerGrid = dim3(num_of_blocks_x, num_of_blocks_y, 1);
        threadsPerBlock = dim3(num_of_threads_x, num_of_threads_y);
    }
#endif  //USE_X_DIMENSIONS_ONLY
    
    __wchar_t* rotate_90_status1 = rotate_90(device_inputData1, device_outputData1, uchar_subPixelType, 
        input_image_width, input_image_height, is_clockwise,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);


    

    __wchar_t* rotate_90_status2 = rotate_90(device_inputData2, device_outputData2, ushort_subPixelType, 
        input_image_width, input_image_height, is_clockwise,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    unsigned char* outputData1 = (unsigned char*)malloc(device_outputData_num_of_bytes1);
    HANDLE_ERROR(cudaMemcpy(outputData1, device_outputData1, device_outputData_num_of_bytes1, cudaMemcpyDeviceToHost));
    image2_uchar.data = outputData1;

    // Copy output vector from GPU buffer to host memory.
    unsigned char* outputData2 = (unsigned char*)malloc(device_outputData_num_of_bytes2);
    HANDLE_ERROR(cudaMemcpy(outputData2, device_outputData2, device_outputData_num_of_bytes2, cudaMemcpyDeviceToHost));
    image2_ushort.data = outputData2;

    HANDLE_ERROR(cudaFree(device_inputData1));
    HANDLE_ERROR(cudaFree(device_inputData2));
    HANDLE_ERROR(cudaFree(device_outputData1));
    HANDLE_ERROR(cudaFree(device_outputData2));

#else //USE_CUDA
    unsigned int host_outputData_num_of_elements = image1_uchar.rows * image1_uchar.cols;
    size_t host_outputData_num_of_bytes1 = host_outputData_num_of_elements * sizeof(unsigned char);
    unsigned char* host_outputData1 = (unsigned char*)(malloc(host_outputData_num_of_bytes1));
    blocksPerGrid = block_and_grid_dims->blocksPerGrid;
    threadsPerBlock = block_and_grid_dims->threadsPerBlock;

    __wchar_t* rotate_90_status1 = rotate_90_cpu(image1_uchar.data, host_outputData1, uchar_subPixelType,
        input_image_width, input_image_height, is_clockwise,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);
    image2_uchar.data = host_outputData1;


    size_t host_outputData_num_of_bytes2 = host_outputData_num_of_elements * sizeof(unsigned short);
    unsigned char* host_outputData2 = (unsigned char*)(malloc(host_outputData_num_of_bytes2));

    __wchar_t* rotate_90_status2 = rotate_90_cpu(image1_ushort.data, host_outputData2, ushort_subPixelType,
        input_image_width, input_image_height, is_clockwise,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);
    image2_ushort.data = host_outputData2;
#endif //USE_CUDA

    auto stop_time_cpu = std::chrono::high_resolution_clock::now();
    auto duration_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_cpu - start_time_cpu);
    auto num_of_milliseconds = duration_milliseconds.count();
    std::cout << "Rotate_90 image time = " << num_of_milliseconds << " milliseconds. " << std::endl;


    if (read_image_from_file == true)
    {
        double scale_factor = 0.25;
        cv::Mat resized_image1_uchar = calc_resized_image(image1_uchar, scale_factor);
        cv::Mat resized_image2_uchar = calc_resized_image(image2_uchar, scale_factor);
        cv::Mat resized_image1_ushort = calc_resized_image(image1_ushort, scale_factor);
        cv::Mat resized_image2_ushort = calc_resized_image(image2_ushort, scale_factor);

        cv::imshow("resized_image1_uchar", resized_image1_uchar);
        cv::imshow("resized_image2_uchar", resized_image2_uchar);

        cv::imshow("resized_image1_ushort", resized_image1_ushort);
        cv::imshow("resized_image2_ushort", resized_image2_ushort);

        //cv::imshow("image1_float", image1_float);
        //cv::imshow("image2_float", image2_float);
    }
    else
    {
        bool is_hex = true;
        print_pixels("image1_uchar", image1_uchar.data, image1_uchar.rows, image1_uchar.cols, PixelType::UCHAR, is_hex);
        print_pixels("image2_uchar", image2_uchar.data, image2_uchar.rows, image2_uchar.cols, PixelType::UCHAR, is_hex);

        print_pixels("image1_ushort", image1_ushort.data, image1_ushort.rows, image1_ushort.cols, PixelType::USHORT, is_hex);
        print_pixels("image2_ushort", image2_ushort.data, image2_ushort.rows, image2_ushort.cols, PixelType::USHORT, is_hex);

        //print_pixels("image1_float", image1_ushort.data, image1_ushort.rows, image1_ushort.cols, PixelType::FLOAT);
        //print_pixels("image2_float", image2_ushort.data, image2_ushort.rows, image2_ushort.cols, PixelType::FLOAT);
    }

    int k = cv::waitKey(0); // Wait for a keystroke in the window



    return 0;
}