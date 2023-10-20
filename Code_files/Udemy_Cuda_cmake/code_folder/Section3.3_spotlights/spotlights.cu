// Render several spotlights on an image.
// Uses a two-dimensional grid with a one-dimensional memory layout, so
// performance is not optimal.
// Example for video 3.3.

#include <cmath>
#include <cstdint>
#include <iostream>


// Standard CUDA API functions
#include <cuda_runtime_api.h>

#include "utils.h"




struct light {
  float x;
  float y;
  float radius;
  float brightness;
};

__device__ float clamp(float value) { return value > 1.0f ? 1.0f : value; }

__device__ float light_brightness(float x, float y, unsigned int width,
                                  unsigned int height, const light &light)
{
  float norm_x = x / width;
  float norm_y = y / height;

  float dx = norm_x - light.x;
  float dy = norm_y - light.y;
  float distance_squared = dx * dx + dy * dy;
  if (distance_squared > light.radius * light.radius) {
    return 0;
  }
  float distance = sqrtf(distance_squared);

  float scaled_distance = distance / light.radius;
  if (scaled_distance > 0.8) {
    return (1.0f - (scaled_distance - 0.8f) * 5.0f) * light.brightness;
  } else {
    return light.brightness;
  }
}

//__global__ void spotlights(const image source, image dest, unsigned int width,
//                           unsigned int height, float ambient, light light1,
//                           light light2, light light3, light light4)
//{
//  int x = blockIdx.x * blockDim.x + threadIdx.x;
//  int y = blockIdx.y * blockDim.y + threadIdx.y;
//  if (x >= width || y >= height) return;
//
//  int index = y * width + x;
//
//  float brightness = ambient + light_brightness(x, y, width, height, light1) +
//                     light_brightness(x, y, width, height, light2) +
//                     light_brightness(x, y, width, height, light3) +
//                     light_brightness(x, y, width, height, light4);
//
//  dest.red[index] = clamp(source.red[index] * brightness);
//  dest.green[index] = clamp(source.green[index] * brightness);
//  dest.blue[index] = clamp(source.blue[index] * brightness);
//}

__global__ void spotlights(const image source, image dest, unsigned int width,
	unsigned int height, float ambient, light light1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int index = y * width + x;

	float brightness = ambient + light_brightness(x, y, width, height, light1);

	dest.red[index] = clamp(source.red[index] * brightness);
	dest.green[index] = clamp(source.green[index] * brightness);
	dest.blue[index] = clamp(source.blue[index] * brightness);
}

int main(int argc, char **argv)
{
	char command_name[] = "spotlights";
	//char input_image_name[] = "../bridge.ppm";
	//going back from this folder: ./build/code_folder/Section3.3_spotlights/
	char input_image_full_path[] = "../../../images/bridge.ppm";
	//char input_image_full_path[] = "../../../code_folder/Section3.3_spotlights/images/unity_frame.jpg";
	char output_image_full_path[] = "../../../images/output_unity_frame.ppm";
	char* my_argv[3];
	my_argv[0] = command_name;
	my_argv[1] = input_image_full_path;
	my_argv[2] = output_image_full_path;
	int my_argc = 3;

	auto params = set_up_test_planar(my_argc, my_argv);

	float light1_x = 0.5f;
	float light1_y = 0.5f;
	float light1_radius = 0.3f;
	float light1_brightness = 1.0;
  light light1 = { light1_x, light1_y, light1_radius, light1_brightness };
  //light light2 = {0.25, 0.2, 0.075, 2.0};
  //light light3 = {0.5, 0.5, 0.3, 0.3};
  //light light4 = {0.7, 0.65, 0.15, 0.8};

  dim3 BLOCK_DIM(32, 16);
  dim3 grid_dim((params.width + BLOCK_DIM.x - 1) / BLOCK_DIM.x,
                (params.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y);
  float ambient = 0.0f;
  {
    KernelTimer t;
    //spotlights<<<grid_dim, BLOCK_DIM>>>(params.input_image, params.output_image,
    //                                    params.width, params.height, 0.3,
    //                                    light1, light2, light3, light4);
	
	spotlights << <grid_dim, BLOCK_DIM >> > (params.input_image, params.output_image,
		params.width, params.height, ambient, light1);
  }

  finish_test_planar(params);

  return 0;
}
