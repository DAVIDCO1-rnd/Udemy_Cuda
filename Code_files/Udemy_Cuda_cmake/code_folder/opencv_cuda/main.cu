#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <cmath>

#define USE_CUDA

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

__device__ inline int PixelOffset1D(int x, int channel, int pixelSize, int channelSize)
{
    return  x * pixelSize + channel * channelSize;
}

__device__ inline int PixelOffset(int y, int x, int channel, int stride, int pixelSize, int channelSize)
{
    return y * stride + PixelOffset1D(x, channel, pixelSize, channelSize);
}

__device__  inline bool DecodeYXC(int* y, int* x, int* c, int widthImage, int heightImage)
{
    *y = (threadIdx.y) + (blockDim.y) * (blockIdx.y);
    *x = (threadIdx.x) + (blockDim.x) * (blockIdx.x);
    *c = (threadIdx.z);

    return (*y >= 0 && *y < heightImage&&* x >= 0 && *x < widthImage);
}

template<class T> __global__ void build_image_rotated_by_90_degrees_cuda(unsigned char* device_inputData, unsigned char* device_outputData, int* device_input_width, int* device_input_height, int* device_pixel_size, int is_clockwise)
{


    int input_width = device_input_width[0];
    int input_height = device_input_height[0];
    int output_width = input_height;
    int output_height = input_width;
    int pixel_size = device_pixel_size[0];

    int i = blockIdx.x;
    

    while (i < input_width)
    {
        int j = threadIdx.x;
        while (j < input_height)
        {
            int current_index_input_data = pixel_size * (i * input_height + j);
            int current_index_output_data;
            if (is_clockwise == 1)
            {
                current_index_output_data = pixel_size * ((input_height - j - 1) * input_width + i); //Clockwise
            }
            else
            {
                current_index_output_data = pixel_size * (j * input_width + input_width - 1 - i); //CounterClockwise
            }
            T pixel_value = *(T*)(device_inputData + current_index_output_data);
            *((T*)(device_outputData + current_index_input_data)) = pixel_value;
            j += blockDim.x;
        }
        i += gridDim.x;
    }
}
#endif //USE_CUDA

template <class T>
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
            T pixel_value = *(T*)(inputData + current_index_output_data);
            *((T*)(outputData + current_index_input_data)) = pixel_value;
            
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


class BlockAndGridDimensions {
private:
    int gridSizes[2];
    int blockSizes[3];

public:
    BlockAndGridDimensions(int block_sizes[3], int grid_sizes[2]) {
        for (int i = 0; i < 2; ++i) {
            gridSizes[i] = grid_sizes[i];
        }

        for (int i = 0; i < 3; ++i) {
            blockSizes[i] = block_sizes[i];
        }
    }
};

//c++ code:
BlockAndGridDimensions* CalculateBlockAndGridDimensions(int channels, int width, int height)
{
    cudaDeviceProp  prop;
    int device_index = 0; //For now I assume there's only one GPu device
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device_index));
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlockSize = maxThreadsPerBlock / 2;

    int blockSize[3];
    int gridSize[2];

    // Calculate optimal block size, depends on the number of channels in picture
    if (width * height * channels < maxBlockSize)
    {
        blockSize[0] = width;
        blockSize[1] = height;
    }
    else
    {
        int warpSize = prop.warpSize;
        float dWarp = warpSize / (float)channels;
        int maxSize = (int)(maxBlockSize / (float)channels);

        if (width <= maxSize)
            blockSize[0] = width;
        else
        {
            float threadsX = 0.0f;
            while (threadsX < maxSize)
            {
                threadsX += dWarp;

            }
            blockSize[0] = (int)threadsX;
        }
        blockSize[1] = maxSize / blockSize[0];
        if (blockSize[1] == 0)
        {
            blockSize[1] = 1;
        }
    }

    //block size 3rd dimension is always the number of channels.
    blockSize[2] = channels;

    //calculate grid size. (number of necessary blocks to cover the whole picture) 
    gridSize[0] = (int)ceil((double)width / blockSize[0]);
    gridSize[1] = (int)ceil((double)height / blockSize[1]);

    BlockAndGridDimensions* block_and_grid_dimensions = new BlockAndGridDimensions(blockSize, gridSize);
    return block_and_grid_dimensions;

    //return new BlockAndGridDimensions(
    //    blockSize,
    //    gridSize
    //);
}

//c# code:
//public static BlockAndGridDimensions CalculateBlockAndGridDimensions(int channels, int width, int height)
//{
//
//    var maxBlockSize = DeviceProperties.deviceThreadsPerBlock / 2;
//
//
//    var blockSize = new int[3];
//    var gridSize = new int[2];
//
//    // Calculate optimal block size, depends on the number of channels in picture
//    if (width * height * channels < maxBlockSize)
//    {
//        blockSize[0] = width;
//        blockSize[1] = height;
//    }
//    else
//    {
//        var dWarp = DeviceProperties.deviceWarpSize / (float)channels;
//        var maxSize = (int)(maxBlockSize / (float)channels);
//
//        if (width <= maxSize)
//            blockSize[0] = width;
//        else
//        {
//            var threadsX = 0.0f;
//            while (threadsX < maxSize)
//            {
//                threadsX += dWarp;
//
//            }
//            blockSize[0] = (int)threadsX;
//        }
//        blockSize[1] = maxSize / blockSize[0];
//        if (blockSize[1] == 0)
//        {
//            blockSize[1] = 1;
//        }
//    }
//
//    //block size 3rd dimension is always the number of channels.
//    blockSize[2] = channels;
//
//    //calculate grid size. (number of necessary blocks to cover the whole picture) 
//    gridSize[0] = (int)Math.Ceiling((double)width / blockSize[0]);
//    gridSize[1] = (int)Math.Ceiling((double)height / blockSize[1]);
//
//    return new BlockAndGridDimensions(
//        blockSize,
//        gridSize
//    );
//}

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
    
    int blocksPerGrid = 256;    //drimDim is two-dimensional
    int threadsPerBlock = 256;  //blockDim is three-dimensional

    
    //int threadsPerBlock = image_height;
    //int blocksPerGrid = (image_height * image_width + threadsPerBlock - 1) / threadsPerBlock;

    BlockAndGridDimensions* block_and_grid_dims = CalculateBlockAndGridDimensions(num_of_channels, image_width, image_height);

    int is_clockwise = 1;
    build_image_rotated_by_90_degrees_cuda<unsigned char> << < blocksPerGrid, threadsPerBlock >> > (device_inputData1, device_outputData1, device_input_width, device_input_height, device_uchar_pixel_size, is_clockwise);
    build_image_rotated_by_90_degrees_cuda<unsigned short> << < blocksPerGrid, threadsPerBlock >> > (device_inputData2, device_outputData2, device_input_width, device_input_height, device_ushort_pixel_size, is_clockwise);

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
        double scale_factor = 0.35;
        cv::Mat resized_image1_uchar = calc_resized_image(image1_uchar, scale_factor);
        cv::Mat resized_image2_uchar = calc_resized_image(image2_uchar, scale_factor);
        cv::Mat resized_image1_ushort = calc_resized_image(image1_ushort, scale_factor);
        cv::Mat resized_image2_ushort = calc_resized_image(image2_ushort, scale_factor);
        
        cv::imshow("resized_image1_uchar", resized_image1_uchar);
        cv::imshow("resized_image2_uchar", resized_image2_uchar);

        //cv::imshow("resized_image1_ushort", resized_image1_ushort);
        //cv::imshow("resized_image2_ushort", resized_image2_ushort);

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