#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>

#include <iostream>


void print_pixels(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2)
{
    printf("%s\n", matrix_name.c_str());
    for (int i = 0; i < dimension1; i++)
    {
        for (int j = 0; j < dimension2; j++)
        {
            size_t current_index = j * dimension1 + i;
            unsigned char current_val = pixelData[current_index];
            printf("%d, ", current_val);
        }
        printf("\n");
    }
    printf("\n\n");
}

int main()
{
    //going back from this folder: ./build/code_folder/Section3.3_spotlights/
    std::string image_path = "../../../code_folder/opencv_cuda/images/grayscale.png";
    //cv::Mat image = cv::imread(image_path);
    //if (image.empty())
    //{
    //    std::cout << "Could not read the image: " << image_path << std::endl;
    //    return 1;
    //}

    const int width3 = 4;
    const int height3 = 3;

    uchar image_data[height3][width3] = {
       {10, 20, 30, 40},
       {50, 60, 70, 80},
       {90, 100, 110, 120}
    };

    cv::Mat image1(height3, width3, CV_8UC1); // 3 rows, 4 columns, 8-bit single-channel (grayscale)
    for (int y = 0; y < image1.rows; ++y) {
        for (int x = 0; x < image1.cols; ++x) {
            image1.at<uchar>(y, x) = static_cast<uchar>(image_data[y][x]);
        }
    }

    cv::imwrite(image_path, image1);
    cv::Mat image2 = cv::imread(image_path);







    unsigned char* pixelData1 = image1.data;
    int height1 = image1.rows;
    int width1 = image1.cols;

    unsigned char* pixelData2 = image1.data;
    int height2 = image2.rows;
    int width2 = image2.cols;

    print_pixels("pixelData1", pixelData1, height1, width1);
    print_pixels("pixelData2", pixelData2, height2, width2);

    //cv::imshow("Display window", image);
    //int k = cv::waitKey(0); // Wait for a keystroke in the window
    //if (k == 's')
    //{
    //    cv::imwrite("starry_night.png", image);
    //}


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