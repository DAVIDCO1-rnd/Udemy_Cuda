#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_utils.cuh"
#include "opencv_utils.h"

#define USE_CUDA

bool read_image_from_file = true;
const int height = 3;
const int width = 5;


void print_single_val(unsigned char* pixelData, int i)
{
    unsigned char current_val = pixelData[i];
    printf("0x%02x, ", current_val);
}


void print_pixels_1D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2)
{
    int pixel_size = 1;
    printf("%s as 1D array\n", matrix_name.c_str());
    for (int i = 0; i < pixel_size * dimension1 * dimension2; i+=pixel_size)
    {
        print_single_val(pixelData, i);
    }
    printf("\n\n");
}

void print_pixels_2D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2)
{
    int pixel_size = 1;
    printf("%s as 2D array\n", matrix_name.c_str());
    for (int i = 0; i < dimension1; i++)
    {
        for (int j = 0; j < pixel_size * dimension2; j += pixel_size)
        {
            int current_index = i * pixel_size * dimension2 + j;
            print_single_val(pixelData, current_index);
        }
        printf("\n");
    }
    printf("\n\n");
}

void print_pixels(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2)
{
    print_pixels_1D(matrix_name, pixelData, dimension1, dimension2);
    print_pixels_2D(matrix_name, pixelData, dimension1, dimension2);
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


#define DIMENSIONS 512
#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

struct Sphere {
    float   r, b, g;
    float   radius;
    float   x, y, z;
    __device__ float hit(float ox, float oy, float* n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};
#define NUM_OF_SPHERES 20

__constant__ Sphere sphere_object[NUM_OF_SPHERES];

__global__ void kernel(unsigned char* ptr) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIMENSIONS / 2);
    float   oy = (y - DIMENSIONS / 2);

    float   r = 0, g = 0, b = 0;
    float   maxz = -INF;
    for (int i = 0; i < NUM_OF_SPHERES; i++) {
        float   n;
        float   t = sphere_object[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = sphere_object[i].r * fscale;
            g = sphere_object[i].g * fscale;
            b = sphere_object[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}   
#endif //USE_CUDA


int main()
{
    int ticks = 0;
    int num_of_frames = 1;
    int width = DIMENSIONS;
    int height = DIMENSIONS;
    //going back from this folder: ./build/code_folder/Section3.3_spotlights/
    //std::string image_path = "../../../images/balloons.jpg";
    cv::Mat image1_uchar;
    cv::Mat image1_ushort;
    cv::Mat image1_float;
    if (read_image_from_file == true)
    {
        //cv::Mat rgb_image1 = cv::imread(image_path);        
        //cv::cvtColor(rgb_image1, image1_uchar, cv::COLOR_BGR2GRAY);

        image1_uchar = cv::Mat(width, height, CV_8UC1, cv::Scalar(0));
        image1_uchar.convertTo(image1_ushort, CV_16UC1, 256);
        image1_uchar.convertTo(image1_float, CV_32FC1, 65536);
    }



    cv::Mat image2_uchar(image1_uchar.cols, image1_uchar.rows, CV_8UC4);
    //cv::Mat image2_ushort(image1_ushort.cols, image1_ushort.rows, CV_16UC1);
    //cv::Mat image2_float(image1_float.cols, image1_float.rows, CV_32FC1);

    for (int i = 0; i < num_of_frames; i++)
    {
        ticks = ticks + 1;
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
        size_t device_outputData_num_of_bytes1 = device_outputData_num_of_elements * sizeof(unsigned char) * 4;
        HANDLE_ERROR(cudaMalloc((void**)&device_outputData1, device_outputData_num_of_bytes1));

        //size_t device_outputData_num_of_bytes2 = device_outputData_num_of_elements * sizeof(unsigned short);
        //HANDLE_ERROR(cudaMalloc((void**)&device_outputData2, device_outputData_num_of_bytes2));





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
        int uchar_pixel_size = 1;
        HANDLE_ERROR(cudaMemcpy(device_uchar_pixel_size, &(uchar_pixel_size), device_uchar_pixel_size_bytes, cudaMemcpyHostToDevice));

        //create device_ushort_pixel_size
        int* device_ushort_pixel_size = NULL;
        size_t device_ushort_pixel_size_bytes = sizeof(int);
        HANDLE_ERROR(cudaMalloc((void**)&device_ushort_pixel_size, device_ushort_pixel_size_bytes));



        int image_height = image1_uchar.rows;
        int image_width = image1_uchar.cols;
        int num_of_channels = 1;

        //int blocksPerGrid = 256;    //dridDim is two-dimensional
        //int threadsPerBlock = 256;  //blockDim is three-dimensional


        cudaDeviceProp  prop;
        int device_index = 0; //For now I assume there's only one GPu device
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device_index));
        int maxThreadsPerBlock = prop.maxThreadsPerBlock;

        //int threadsPerBlock = std::min(image_height, maxThreadsPerBlock);
        //int blocksPerGrid = (image_height * image_width + threadsPerBlock - 1) / threadsPerBlock;

        int num_of_threads_x = 16;
        int num_of_threads_y = 16;

        int num_of_blocks_x = image_width / num_of_threads_x;
        int num_of_blocks_y = image_height / num_of_threads_y;

        dim3 blocksPerGrid(num_of_blocks_x, num_of_blocks_y);
        dim3 threadsPerBlock(num_of_threads_x, num_of_threads_y);

        BlockAndGridDimensions* block_and_grid_dims = CalculateBlockAndGridDimensions(num_of_channels, image_width, image_height);


        // allocate temp memory, initialize it, copy to constant
        // memory on the GPU, then free our temp memory
        Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * NUM_OF_SPHERES);
        for (int i = 0; i < NUM_OF_SPHERES; i++) {
            temp_s[i].r = rnd(1.0f);
            temp_s[i].g = rnd(1.0f);
            temp_s[i].b = rnd(1.0f);
            temp_s[i].x = rnd(1000.0f) - 500;
            temp_s[i].y = rnd(1000.0f) - 500;
            temp_s[i].z = rnd(1000.0f) - 500;
            temp_s[i].radius = rnd(100.0f) + 20;
        }
        HANDLE_ERROR(cudaMemcpyToSymbol(sphere_object, temp_s, sizeof(Sphere) * NUM_OF_SPHERES));
        free(temp_s);

        dim3    grids(DIMENSIONS / 16, DIMENSIONS / 16);
        dim3    threads(16, 16);
        kernel << <grids, threads >> > (device_outputData1);

        // Check for any errors launching the kernel
        HANDLE_ERROR(cudaGetLastError());


        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Copy output vector from GPU buffer to host memory.
        unsigned char* outputData1 = (unsigned char*)malloc(device_outputData_num_of_bytes1);
        HANDLE_ERROR(cudaMemcpy(outputData1, device_outputData1, device_outputData_num_of_bytes1, cudaMemcpyDeviceToHost));

        image2_uchar.data = outputData1;

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
            //double scale_factor = 0.25;
            //cv::Mat resized_image1_uchar = calc_resized_image(image1_uchar, scale_factor);
            //cv::Mat resized_image2_uchar = calc_resized_image(image2_uchar, scale_factor);
            //cv::Mat resized_image1_ushort = calc_resized_image(image1_ushort, scale_factor);
            //cv::Mat resized_image2_ushort = calc_resized_image(image2_ushort, scale_factor);

            //cv::imshow("resized_image1_uchar", resized_image1_uchar);
            cv::imshow("image2_uchar", image2_uchar);
        }
        else
        {
            print_pixels("image1_uchar", image1_uchar.data, image1_uchar.rows, image1_uchar.cols);
            print_pixels("image2_uchar", image2_uchar.data, image2_uchar.rows, image2_uchar.cols);
        }

        int k = cv::waitKey(0); // Wait for a keystroke in the window
    }





    getchar();



    return 0;
}

