// Utility functions for example programs.

#include "stb_image.h"
#include <assert.h>
//#include <getopt.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

#include <cuda_runtime_api.h>

#include "utils.h"

using std::string;

const unsigned int HEADER_SIZE = 0x40;
const unsigned int CHANNELS = 3;

bool loadJPG(char *filePath, pixel **data, unsigned int *w, unsigned int *h)
{
	int width, height, textureNumOfChannels;
	unsigned char *textureData = stbi_load(filePath, &width, &height, &textureNumOfChannels, 0);
	*w = (unsigned int)width;
	*h = (unsigned int)height;
	if (textureData)
	{
		int pixel_count = width * height;
		pixel *pixel_data = static_cast<pixel *>(malloc(pixel_count * sizeof(pixel)));
		int inputCounter = 0;
		int outputCounter = 0;
		float scale = 1.0f / 255.0f;
		for (int i = 0 ; i < width ; i++)
		{
			for (int j = 0 ; j < height ; j++)
			{
				unsigned char red_char = textureData[inputCounter];
				inputCounter++;
								
				unsigned char green_char = textureData[inputCounter];
				inputCounter++;
							
				unsigned char blue_char = textureData[inputCounter];
				inputCounter++;

				float red = (float)red_char * scale;
				float green = (float)green_char * scale;
				float blue = (float)blue_char * scale;

				pixel_data[outputCounter].red = red;
				pixel_data[outputCounter].green = green;
				pixel_data[outputCounter].blue = blue;
				pixel_data[outputCounter].alpha = 1.0f;
				outputCounter++;
			}
		}
		*data = pixel_data;
		return true;
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
		return false;
	}
}

bool loadPPM(const char *file, pixel **data, unsigned int *w, unsigned int *h)
{
  FILE *fp = fopen(file, "rb");

  if (!fp) {
    std::cerr << "loadPPM() : failed to open file: " << file << "\n";
    return false;
  }

  // check header
  char header[HEADER_SIZE];

  if (fgets(header, HEADER_SIZE, fp) == nullptr) {
    std::cerr << "loadPPM(): reading header returned NULL\n";
    return false;
  }

  if (strncmp(header, "P6", 2)) {
    std::cerr << "unsupported image format\n";
    return false;
  }

  // parse header, read maxval, width and height
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int maxval = 0;
  unsigned int i = 0;

  while (i < 3) {
    if (fgets(header, HEADER_SIZE, fp) == NULL) {
      std::cerr << "loadPPM() : reading PPM header returned NULL" << std::endl;
      return false;
    }

    if (header[0] == '#') {
      continue;
    }

    if (i == 0) {
      i += sscanf(header, "%u %u %u", &width, &height, &maxval);
    } else if (i == 1) {
      i += sscanf(header, "%u %u", &height, &maxval);
    } else if (i == 2) {
      i += sscanf(header, "%u", &maxval);
    }
  }

  size_t pixel_count = width * height;
  size_t data_size = sizeof(unsigned char) * pixel_count * CHANNELS;
  unsigned char *raw_data = static_cast<unsigned char *>(malloc(data_size));
  *w = width;
  *h = height;

  // read and close file
  if (fread(raw_data, sizeof(unsigned char), pixel_count * CHANNELS, fp) == 0) {
    std::cerr << "loadPPM() read data returned error.\n";
  }
  fclose(fp);

  pixel *pixel_data = static_cast<pixel *>(malloc(pixel_count * sizeof(pixel)));
  float scale = 1.0f / 255.0f;
  for (int i = 0; i < pixel_count; i++) {
    pixel_data[i].red = raw_data[3 * i] * scale;
    pixel_data[i].green = raw_data[3 * i + 1] * scale;
    pixel_data[i].blue = raw_data[3 * i + 2] * scale;
  }

  *data = pixel_data;
  free(raw_data);

  return true;
}

void savePPM(const char *file, pixel *data, unsigned int w, unsigned int h)
{
  assert(data != nullptr);
  assert(w > 0);
  assert(h > 0);

  std::fstream fh(file, std::fstream::out | std::fstream::binary);

  if (fh.bad()) {
    std::cerr << "savePPM() : open failed.\n";
    return;
  }

  fh << "P6\n";
  fh << w << "\n" << h << "\n" << 0xff << "\n";

  unsigned int pixel_count = w * h;
  for (unsigned int i = 0; (i < pixel_count) && fh.good(); ++i) {
    fh << static_cast<unsigned char>(data[i].red * 255);
    fh << static_cast<unsigned char>(data[i].green * 255);
    fh << static_cast<unsigned char>(data[i].blue * 255);
  }

  fh.flush();

  if (fh.bad()) {
    std::cerr << "savePPM() : writing data failed.\n";
    return;
  }

  fh.close();
}

test_params set_up_test(int argc, char **argv)
{
  test_params params = {0, 0, nullptr, nullptr, nullptr};

  bool show_help = false;
  for (int i = 1; i < argc; i++) {
    char *current = argv[i];
    if (!strncmp(current, "--", 2)) {
      show_help = true;
      break;
    } else if (params.input_image == nullptr) {
      // Load input
      //pixel *host_image = nullptr;
      //if (!loadPPM(current, &host_image, &params.width, &params.height)) {
      //  exit(1);
      //}

	  pixel *host_image = nullptr;
	  if (!loadJPG(current, &host_image, &params.width, &params.height)) {
		  exit(1);
	  }

	 

      size_t image_size = params.width * params.height * sizeof(pixel);
      cudaCheckError(cudaMalloc(&params.input_image, image_size));
      cudaCheckError(cudaMalloc(&params.output_image, image_size));
      cudaCheckError(cudaMemcpy(params.input_image, host_image, image_size,
                                cudaMemcpyHostToDevice));

    } else if (params.output_file == nullptr) {
      // Save output filename
      params.output_file = current;
    } else {
      show_help = true;
      break;
    }
  }

  if (!params.output_file || !params.input_image) {
    show_help = true;
  }

  if (show_help) {
    std::cout << "Usage: " << argv[0] << " INPUT_FILE OUTPUT_FILE\n";
    exit(1);
  }

  return params;
}

//void savePixels(string filename, char* pixelsData, int PictureWidth, int PictureHeight)
//{
//	FILE *Out = fopen(filename.c_str(), "wb");
//	if (!Out)
//		return;
//	int nSize = PictureWidth * PictureHeight * 3;
//	BITMAPFILEHEADER bitmapFileHeader;
//	BITMAPINFOHEADER bitmapInfoHeader;
//
//	bitmapFileHeader.bfType = 0x4D42;
//	bitmapFileHeader.bfSize = nSize;
//	bitmapFileHeader.bfReserved1 = 0;
//	bitmapFileHeader.bfReserved2 = 0;
//	bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
//
//	bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
//	bitmapInfoHeader.biWidth = PictureWidth - 1;
//	bitmapInfoHeader.biHeight = PictureHeight - 1;
//	bitmapInfoHeader.biPlanes = 1;
//	bitmapInfoHeader.biBitCount = 24;
//	bitmapInfoHeader.biCompression = BI_RGB;
//	bitmapInfoHeader.biSizeImage = 0;
//	bitmapInfoHeader.biXPelsPerMeter = 0; // ?
//	bitmapInfoHeader.biYPelsPerMeter = 0; // ?
//	bitmapInfoHeader.biClrUsed = 0;
//	bitmapInfoHeader.biClrImportant = 0;
//
//	fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, Out);
//	fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, Out);
//	fwrite(pixelsData, nSize, 1, Out);
//	fclose(Out);
//}

void finish_test(const test_params &params)
{
  std::unique_ptr<pixel[]> host_image(new pixel[params.width * params.height]);

  cudaCheckError(cudaMemcpy(host_image.get(), params.output_image,
                            params.width * params.height * sizeof(pixel),
                            cudaMemcpyDeviceToHost));
  if (params.input_image) {
    cudaCheckError(cudaFree(params.input_image));
  }
  if (params.output_image) {
    cudaCheckError(cudaFree(params.output_image));
  }

  //savePixels(params.output_file, host_image.get(), params.width, params.height);
  savePPM(params.output_file, host_image.get(), params.width, params.height);
}

__global__ void unpack_image(image planar, const pixel *packed, int pixel_count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;

  planar.red[index] = packed[index].red;
  planar.green[index] = packed[index].green;
  planar.blue[index] = packed[index].blue;
}

__global__ void pack_image(const image planar, pixel *packed, int pixel_count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;

  packed[index].red = planar.red[index];
  packed[index].green = planar.green[index];
  packed[index].blue = planar.blue[index];
}

image malloc_image(int pixel_count)
{
  image result;
  cudaCheckError(cudaMalloc(&result.red, pixel_count * sizeof(float)));
  cudaCheckError(cudaMalloc(&result.green, pixel_count * sizeof(float)));
  cudaCheckError(cudaMalloc(&result.blue, pixel_count * sizeof(float)));

  return result;
}

void free_image(const image &img)
{
  cudaCheckError(cudaFree(img.red));
  cudaCheckError(cudaFree(img.green));
  cudaCheckError(cudaFree(img.blue));
}

const int BLOCK_SIZE = 128;

test_params_planar set_up_test_planar(int argc, char **argv)
{
  test_params params1 = set_up_test(argc, argv);
  test_params_planar params = {
      params1.width, params1.height, {}, {}, params1.output_file};

  int pixel_count = params.width * params.height;
  params.input_image = malloc_image(pixel_count);
  params.output_image = malloc_image(pixel_count);

  int n_blocks = (pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unpack_image<<<n_blocks, BLOCK_SIZE>>>(params.input_image,
                                         params1.input_image, pixel_count);

  cudaCheckError(cudaFree(params1.input_image));
  params1.input_image = nullptr;

  return params;
}

void finish_test_planar(const test_params_planar &params)
{
  free_image(params.input_image);

  test_params params1 = {params.width, params.height, nullptr, nullptr,
                         params.output_file};

  int pixel_count = params.width * params.height;
  cudaCheckError(
      cudaMalloc(&params1.output_image, pixel_count * sizeof(pixel)));

  int n_blocks = (pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
  pack_image<<<n_blocks, BLOCK_SIZE>>>(params.output_image,
                                       params1.output_image, pixel_count);

  free_image(params.output_image);

  finish_test(params1);
}

KernelTimer::KernelTimer()
{
  cudaCheckError(cudaDeviceSynchronize());
  start = std::chrono::steady_clock::now();
}

KernelTimer::~KernelTimer()
{
  cudaCheckError(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "kernel ran in " << elapsed << " ms\n";
}
