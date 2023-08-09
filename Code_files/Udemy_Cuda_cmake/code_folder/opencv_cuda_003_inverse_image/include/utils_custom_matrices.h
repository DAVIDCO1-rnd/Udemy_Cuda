

enum class PixelType {
    UCHAR = 1,
    USHORT = 2,
    FLOAT = 4
};

void print_single_val(unsigned char* pixelData, int i, PixelType pixel_type);
void print_pixels_1D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type);
void print_pixels_2D(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type);
void print_pixels(std::string matrix_name, unsigned char* pixelData, int dimension1, int dimension2, PixelType pixel_type);

