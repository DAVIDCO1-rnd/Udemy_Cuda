#include <iostream>
#include "utils_custom_matrices.h"

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
    for (int i = 0; i < pixel_size * dimension1 * dimension2; i += pixel_size)
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



