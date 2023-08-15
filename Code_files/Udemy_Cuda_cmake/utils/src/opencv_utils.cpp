#include "opencv_utils.h"

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