#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

const int height = 5;
const int width = 3;

enum class PixelType {
    UCHAR = 1,
    USHORT = 2,
    FLOAT = 4
};

void print_single_val(unsigned char* pixelData, int i, PixelType pixel_type, bool is_hex);
void print_pixels_1D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type, bool is_hex);
void print_pixels_2D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type, bool is_hex);
void print_pixels(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type, bool is_hex);
cv::Mat build_image_from_data(PixelType pixel_type, int width, int height);

