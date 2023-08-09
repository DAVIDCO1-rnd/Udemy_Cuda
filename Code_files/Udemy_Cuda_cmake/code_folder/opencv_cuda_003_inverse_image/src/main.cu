#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <cmath>

#include "Inverse.cuh"
#include "utils_custom_matrices.h"

#define USE_CUDA
//#define USE_X_DIMENSIONS_ONLY


#ifdef USE_CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#endif //USE_CUDA

bool read_image_from_file = false;
const int height = 3;
const int width = 5;

#ifndef USE_X_DIMENSIONS_ONLY
enum class ThreadsAndBlocksCalculations {
    Use_optimal_function = 0,
    Use_threads_as_warp_size = 1
};
#endif //USE_X_DIMENSIONS_ONLY




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


#ifdef USE_CUDA
class BlockAndGridDimensions {
public:
    dim3 blocksPerGrid;
    dim3 threadsPerBlock;
    BlockAndGridDimensions(dim3 block_sizes, dim3 grid_sizes) {
        blocksPerGrid = grid_sizes;
        threadsPerBlock = block_sizes;
    }
};

BlockAndGridDimensions* CalculateBlockAndGridDimensions(int channels, int width, int height)
{
    cudaDeviceProp  prop;
    int device_index = 0; //For now I assume there's only one GPu device
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device_index));
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlockSize = maxThreadsPerBlock / 2;

    dim3 blockSize;
    dim3 gridSize;

    // Calculate optimal block size, depends on the number of channels in picture
    if (width * height * channels < maxBlockSize)
    {
        blockSize.x = width;
        blockSize.y = height;
    }
    else
    {
        int warpSize = prop.warpSize;
        float dWarp = warpSize / (float)channels;
        int maxSize = (int)(maxBlockSize / (float)channels);

        if (width <= maxSize)
            blockSize.x = width;
        else
        {
            float threadsX = 0.0f;
            while (threadsX < maxSize)
            {
                threadsX += dWarp;

            }
            blockSize.x = (int)threadsX;
        }
        blockSize.y = maxSize / blockSize.x;
        if (blockSize.y == 0)
        {
            blockSize.y = 1;
        }
    }

    //block size 3rd dimension is always the number of channels.
    blockSize.z = channels;

    //calculate grid size. (number of necessary blocks to cover the whole picture) 
    gridSize.x = (int)ceil((double)width / blockSize.x);
    gridSize.y = (int)ceil((double)height / blockSize.y);

    BlockAndGridDimensions* block_and_grid_dimensions = new BlockAndGridDimensions(blockSize, gridSize);
    return block_and_grid_dimensions;
}
#endif //USE_CUDA

cv::Mat calc_resized_image(cv::Mat image, double scale_factor)
{

    // Calculate the new dimensions based on the scale factor
    int newWidth = static_cast<int>(image.cols * scale_factor);
    int newHeight = static_cast<int>(image.rows * scale_factor);

    // Create a new image with the scaled dimensions
    cv::Mat scaledImage;

    // Resize the image using the resize function
    cv::resize(image, scaledImage, cv::Size(newWidth, newHeight), cv::INTER_LINEAR);

    return scaledImage;
}

int main()
{
#ifndef USE_X_DIMENSIONS_ONLY
    ThreadsAndBlocksCalculations threads_and_blocks_calculations = ThreadsAndBlocksCalculations::Use_optimal_function;
#endif //USE_X_DIMENSIONS_ONLY


    //going back from this folder: ./build/code_folder/Section3.3_spotlights/
    std::string image_path = "../../../images/balloons.jpg";
    cv::Mat image1_uchar;
    cv::Mat image1_ushort;
    cv::Mat image1_float;
    if (read_image_from_file == true)
    {
        cv::Mat rgb_image1 = cv::imread(image_path);
        cv::cvtColor(rgb_image1, image1_uchar, cv::COLOR_BGR2GRAY);
        cv::vconcat(image1_uchar, image1_uchar, image1_uchar);
        cv::hconcat(image1_uchar, image1_uchar, image1_uchar);
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



    cv::Mat image2_uchar(image1_uchar.rows, image1_uchar.cols, CV_8UC1);
    cv::Mat image2_ushort(image1_ushort.rows, image1_ushort.cols, CV_16UC1);
    cv::Mat image2_float(image1_float.rows, image1_float.cols, CV_32FC1);


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

    //create device_input_width
    int* device_input_width = NULL;
    size_t device_input_width_bytes = sizeof(int);
    HANDLE_ERROR(cudaMalloc((void**)&device_input_width, device_input_width_bytes));
    HANDLE_ERROR(cudaMemcpy(device_input_width, &(image1_uchar.cols), device_input_width_bytes, cudaMemcpyHostToDevice));


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create device_input_height
    int* device_input_height = NULL;
    size_t device_input_height_bytes = sizeof(int);
    HANDLE_ERROR(cudaMalloc((void**)&device_input_height, device_input_height_bytes));
    HANDLE_ERROR(cudaMemcpy(device_input_height, &(image1_uchar.rows), device_input_height_bytes, cudaMemcpyHostToDevice));



    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //create device_uchar_pixel_size
    int* device_uchar_pixel_size = NULL;
    size_t device_uchar_pixel_size_bytes = sizeof(int);
    HANDLE_ERROR(cudaMalloc((void**)&device_uchar_pixel_size, device_uchar_pixel_size_bytes));
    int uchar_pixel_size = (int)PixelType::UCHAR;
    HANDLE_ERROR(cudaMemcpy(device_uchar_pixel_size, &(uchar_pixel_size), device_uchar_pixel_size_bytes, cudaMemcpyHostToDevice));

    //create device_ushort_pixel_size
    int* device_ushort_pixel_size = NULL;
    size_t device_ushort_pixel_size_bytes = sizeof(int);
    HANDLE_ERROR(cudaMalloc((void**)&device_ushort_pixel_size, device_ushort_pixel_size_bytes));

    int ushort_pixel_size = (int)PixelType::USHORT;
    HANDLE_ERROR(cudaMemcpy(device_ushort_pixel_size, &(ushort_pixel_size), device_ushort_pixel_size_bytes, cudaMemcpyHostToDevice));

    int image_height = image1_uchar.rows;
    int image_width = image1_uchar.cols;
    int num_of_channels = 1;

    //int blocksPerGrid = 256;    //dridDim is two-dimensional
    //int threadsPerBlock = 256;  //blockDim is three-dimensional


    //cudaDeviceProp  prop;
    //int device_index = 0; //For now I assume there's only one GPu device
    //HANDLE_ERROR(cudaGetDeviceProperties(&prop, device_index));
    //int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    //int threadsPerBlock = std::min(image_height, maxThreadsPerBlock);
    //int blocksPerGrid = (image_height * image_width + threadsPerBlock - 1) / threadsPerBlock;

    int num_of_threads_x = 32;
    int num_of_threads_y = 32;

    int num_of_blocks_x = (image_width + num_of_threads_x - 1) / num_of_threads_x;
    int num_of_blocks_y = (image_height + num_of_threads_y - 1) / num_of_threads_y;

    dim3 blocksPerGrid;
    dim3 threadsPerBlock;

#ifdef USE_X_DIMENSIONS_ONLY
    blocksPerGrid = dim3(256, 1, 1);
    threadsPerBlock = dim3(256, 1, 1);
#else //USE_X_DIMENSIONS_ONLY
    if (threads_and_blocks_calculations == ThreadsAndBlocksCalculations::Use_optimal_function)
    {
        BlockAndGridDimensions* block_and_grid_dims = CalculateBlockAndGridDimensions(num_of_channels, image_width, image_height);
        blocksPerGrid = block_and_grid_dims->blocksPerGrid;
        threadsPerBlock = block_and_grid_dims->threadsPerBlock;
    }
    else if (threads_and_blocks_calculations == ThreadsAndBlocksCalculations::Use_threads_as_warp_size)
    {
        blocksPerGrid = dim3(num_of_blocks_x, num_of_blocks_y, 1);
        threadsPerBlock = dim3(num_of_threads_x, num_of_threads_y);
    }
#endif  //USE_X_DIMENSIONS_ONLY

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    unsigned char max_val_uchar = 255;
    int alphaChannelNum = -1;
    int uchar_channelSize = 1;
    int input_image_width = image1_uchar.cols;
    int input_image_height = image1_uchar.rows;
    int uchar_strideSourceImage = input_image_width * uchar_pixel_size;
    int uchar_strideResultImage = input_image_width * uchar_pixel_size;
    int uchar_subPixelType = 1;
    //
    //InvertImageKernel<unsigned char> << < blocksPerGrid, threadsPerBlock >> > (device_inputData1, device_outputData1,
    //    max_val_uchar, alphaChannelNum, uchar_pixel_size, uchar_channelSize,
    //    input_image_width, input_image_height,
    //    uchar_strideSourceImage, uchar_strideResultImage);

    int ushort_subPixelType = 2;
    int ushort_channelSize = 2;
    unsigned short max_val_ushort = 65535;
    int ushort_strideSourceImage = input_image_width * ushort_pixel_size;
    int ushort_strideResultImage = input_image_width * ushort_pixel_size;
    //InvertImageKernel<unsigned short> << < blocksPerGrid, threadsPerBlock >> > (device_inputData2, device_outputData2,
    //    max_val_ushort, alphaChannelNum, ushort_pixel_size, ushort_channelSize,
    //    input_image_width, input_image_height,
    //    ushort_strideSourceImage, ushort_strideResultImage);


    __wchar_t* Inverse_status1 = Inverse(device_inputData1, device_outputData1,
        uchar_subPixelType, max_val_uchar, alphaChannelNum, uchar_pixel_size, uchar_channelSize,
        input_image_width, input_image_height, uchar_strideSourceImage, uchar_strideResultImage,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);

    __wchar_t* Inverse_status2 = Inverse(device_inputData2, device_outputData2,
        ushort_subPixelType, max_val_ushort, alphaChannelNum, ushort_pixel_size, ushort_channelSize,
        input_image_width, input_image_height, ushort_strideSourceImage, ushort_strideResultImage,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);



    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("gpu time = milliseconds =%.8f\n", milliseconds);

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
    HANDLE_ERROR(cudaFree(device_input_width));
    HANDLE_ERROR(cudaFree(device_input_height));
    HANDLE_ERROR(cudaFree(device_uchar_pixel_size));
    HANDLE_ERROR(cudaFree(device_ushort_pixel_size));

#endif //USE_CUDA


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