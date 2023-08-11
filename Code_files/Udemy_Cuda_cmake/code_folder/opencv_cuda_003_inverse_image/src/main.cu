#include <string>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>

#include "Inverse.cuh"
#include "Inverse_cpu.cuh"
#include "utils_custom_matrices.h"




#define USE_CUDA
//#define USE_X_DIMENSIONS_ONLY



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


bool read_image_from_file = true;


#ifndef USE_X_DIMENSIONS_ONLY
enum class ThreadsAndBlocksCalculations {
    Use_optimal_function = 0,
    Use_threads_as_warp_size = 1
};
#endif //USE_X_DIMENSIONS_ONLY








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

    __wchar_t* Inverse_status1 = Inverse(device_inputData1, device_outputData1,
        uchar_subPixelType, max_val_uchar, alphaChannelNum, uchar_pixel_size, uchar_channelSize,
        input_image_width, input_image_height, uchar_strideSourceImage, uchar_strideResultImage,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);
    // Copy output vector from GPU buffer to host memory.
    unsigned char* outputData1 = (unsigned char*)malloc(device_outputData_num_of_bytes1);
    HANDLE_ERROR(cudaMemcpy(outputData1, device_outputData1, device_outputData_num_of_bytes1, cudaMemcpyDeviceToHost));
    image2_uchar.data = outputData1;

    

    __wchar_t* Inverse_status2 = Inverse(device_inputData2, device_outputData2,
        ushort_subPixelType, max_val_ushort, alphaChannelNum, ushort_pixel_size, ushort_channelSize,
        input_image_width, input_image_height, ushort_strideSourceImage, ushort_strideResultImage,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    HANDLE_ERROR(cudaDeviceSynchronize());

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

    __wchar_t* Inverse_status1 = Inverse_cpu(image1_uchar.data, host_outputData1,
        uchar_subPixelType, max_val_uchar, alphaChannelNum, uchar_pixel_size, uchar_channelSize,
        input_image_width, input_image_height, uchar_strideSourceImage, uchar_strideResultImage,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);
    image2_uchar.data = host_outputData1;


    size_t host_outputData_num_of_bytes2 = host_outputData_num_of_elements * sizeof(unsigned short);
    unsigned char* host_outputData2 = (unsigned char*)(malloc(host_outputData_num_of_bytes2));
    __wchar_t* Inverse_status2 = Inverse_cpu(image1_ushort.data, host_outputData2,
        ushort_subPixelType, max_val_ushort, alphaChannelNum, ushort_pixel_size, ushort_channelSize,
        input_image_width, input_image_height, ushort_strideSourceImage, ushort_strideResultImage,
        threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
        blocksPerGrid.x, blocksPerGrid.y);
    image2_ushort.data = host_outputData2;
#endif //USE_CUDA

    auto stop_time_cpu = std::chrono::high_resolution_clock::now();
    auto duration_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_cpu - start_time_cpu);
    std::cout << "inverse image time = " << duration_milliseconds.count() << " milliseconds. " << std::endl;


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