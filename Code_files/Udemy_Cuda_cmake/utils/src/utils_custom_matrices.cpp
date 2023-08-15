#include <iostream>
#include "utils_custom_matrices.h"

void print_single_val(unsigned char* pixelData, int i, PixelType pixel_type, bool is_hex)
{
    if (pixel_type == PixelType::UCHAR)
    {
        unsigned char current_val = pixelData[i];
        if (is_hex == true)
        {
            printf("0x%02x, ", current_val);
        }
        else
        {
            printf("%d,\t", current_val);
        }
        
    }
    if (pixel_type == PixelType::USHORT)
    {
        unsigned char sub_pixel1 = pixelData[i + 0];
        unsigned char sub_pixel2 = pixelData[i + 1];
        unsigned short current_val = 0x100 * sub_pixel2 + sub_pixel1;
        if (is_hex == true)
        {
            printf("0x%04x, ", current_val);
        }
        else
        {
            printf("%d, ", current_val);
        }
        
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


void print_pixels_1D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type, bool is_hex)
{
    int pixel_size = (int)pixel_type;
    printf("%s as 1D array\n", matrix_name.c_str());
    for (int i = 0; i < pixel_size * dimension1 * dimension2; i += pixel_size)
    {
        print_single_val(pixelData, i, pixel_type, is_hex);
    }
    printf("\n\n");
}

void print_pixels_2D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type, bool is_hex)
{
    int pixel_size = (int)pixel_type;
    printf("%s as 2D array\n", matrix_name.c_str());
    for (int i = 0; i < dimension1; i++)
    {
        for (int j = 0; j < pixel_size * dimension2; j += pixel_size)
        {
            int current_index = i * pixel_size * dimension2 + j;
            print_single_val(pixelData, current_index, pixel_type, is_hex);
        }
        printf("\n");
    }
    printf("\n\n");
}

void print_pixels(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type, bool is_hex)
{
    print_pixels_1D(matrix_name, pixelData, dimension1, dimension2, pixel_type, is_hex);
    print_pixels_2D(matrix_name, pixelData, dimension1, dimension2, pixel_type, is_hex);
}

cv::Mat build_image_from_data(PixelType pixel_type, int width, int height)
{
    cv::Mat image;
    uchar uchar_current_val = 0;
    ushort ushort_current_val = 0;
    float float_current_val = 0.0f;

    switch (pixel_type)
    {
    case PixelType::UCHAR:        
        image = cv::Mat(height, width, CV_8UC1);        
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                image.at<uchar>(y, x) = uchar_current_val;
                uchar_current_val++;
            }
        }
        break;

    case PixelType::USHORT:        
        image = cv::Mat(height, width, CV_16UC1);        
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                image.at<ushort>(y, x) = ushort_current_val;
                ushort_current_val++;
            }
        }
        break;

    case PixelType::FLOAT:
        image = cv::Mat(height, width, CV_32FC1);        
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                image.at<float>(y, x) = float_current_val;
                float_current_val++;
            }
        }
        break;

    }
    return image;
}



