/* 
this file consits of some function that runs on the GPU only.

all curnel files using this code should include "Utils_device.cu" 

the purpose of the functions are to give accsses to a pixel/channel in a buffer
*/

#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#pragma once

#define PI ( 3.14159265358979323846f )

#define PI_DIVISION 0.318309886183791f // 1/PI 

#define SPECULAR_ALPHA 1.170171f

#define SPECULAR_BETA 246.0f

#define GRAYFLOAT_SIZE 4

#define GRAY16_SIZE 2

#define GRAY8_SIZE 1

#define MIN(a,b) ((a) <= (b) ? (a) : (b))

#define MAX(a,b) ((a) >= (b) ? (a) : (b))

#define ABS(a) ((a) < 0 ? (-a) : (a))

#define SQR(x) ((x) * (x))

#define FRACTION(x) ((x) - (floorf(x)))

#define KM_TO_METER(km) (((float)(km)) * 1000)

#define METER_TO_KM(m) ((float)(m) *  (1E-3f))

#define DEG_TO_RAD(angle) (angle * 0.017453292599433f) 

#define RAD_TO_DEG(angle) (angle * 57.2957795131f) 


// Calculate linear interpolation
__device__ inline float Interpolate1D(float value0, float value1, float du)
{
	return value0 * (1 - du) + value1 * (du);
}

// Calculate bi-linear interpolation
__device__ inline float Interpolate2D(float value00, float value01, float value10, float value11, float du, float dv)
{
	return Interpolate1D(
		Interpolate1D(value00, value01, du),
		Interpolate1D(value10, value11, du),
		dv);

}

// Calculate tri-linear interpolation
__device__ inline float Interpolate3D(float value000, float value001, float value010, float value011, 
							   float value100, float value101, float value110, float value111, 
							   float du, float dv, float dw)
{
	return Interpolate1D(
		Interpolate2D(value000, value001, value010, value011, du, dv),
		Interpolate2D(value100, value101, value110, value111, du, dv),
		dw);

}
__device__ inline bool IsInImage(int y, int x, int height, int width)
{
	return (y >= 0 && x >= 0 && y < height && x < width);
}

 __device__ inline int PixelOffset1D(int x, int channel, int pixelSize, int channelSize)
{
	return  x * pixelSize + channel * channelSize;
}

 __device__ inline int PixelOffset(int y, int x, int channel, int stride, int pixelSize, int channelSize)
{
	return y * stride + PixelOffset1D(x, channel, pixelSize, channelSize);
}

__device__ inline int PixelTextureOffset(int y, int x, int channel, int strideDividedByChannelSize, int pixelSizeDividedByChannelSize)
{
	return y * strideDividedByChannelSize + x * pixelSizeDividedByChannelSize + channel;
}

 template<class T> __device__  inline T* Pixel(void* buffer, int offset)
{
	return (T*)((unsigned char*)buffer + offset);
}

 template<class T, int channels> 
 __device__  inline void SetPixel(T* pixel, T* value)
{
	for( int iChannel = 0; iChannel < channels; iChannel++)
	{
		pixel[iChannel] = value[iChannel];
	}
}

 template<class T> __device__  int findBin(T pixelValue, int pixelType, int numberOfBins)
{

	switch (pixelType)
	{
		case 1: // 8bit (0...0xff)
		{
			return floor(pixelValue * (double)numberOfBins / (double)0xff);
			
		}
		case 2: // 16bit (0...0xffff)
		{
			return floor(pixelValue * (double)numberOfBins / (double)0xfe01);
			
		}
		case 4: // float (0...1.0f)
		{
			return floor(pixelValue * (double)numberOfBins);
			
		}
	}
		return -1;
}

 /**
 * Decode X from thread and block index and dimensions
 * @returns true if pixel is within image row
 **/
 __device__ inline bool DecodeX(int* x, int widthImage)
 {
	*x = (threadIdx.x) + (blockDim.x)*(blockIdx.x);
	return (*x >= 0 && *x < widthImage);
 }

  /**
 * Decode Y from thread and block index and dimensions
 * @returns true if pixel is within image row
 **/
 __device__ inline bool DecodeY(int* y, int heightImage)
 {
	*y = (threadIdx.y) + (blockDim.y)*(blockIdx.y);
	return (*y >= 0 && *y < heightImage);
 }

/**
 * Decode Y and X from thread and block index and dimensions
 * @returns true if pixel is within image
 **/
__device__ inline bool DecodeYX(int* y, int* x, int widthImage, int heightImage)
{
	*y = (threadIdx.y) + (blockDim.y)*(blockIdx.y);
	*x = (threadIdx.x) + (blockDim.x)*(blockIdx.x);

	return (*y >= 0 && *y < heightImage && *x >= 0 && *x < widthImage);
}

/**
 * Decode Y, X and channel from thread and block index and dimensions
 * @returns true if pixel is within image
 **/
 __device__  inline bool DecodeYXC(int* y, int* x, int* c, int widthImage, int heightImage)
{
	*y = (threadIdx.y) + (blockDim.y)*(blockIdx.y);
	*x = (threadIdx.x) + (blockDim.x)*(blockIdx.x);
	*c = (threadIdx.z);
	
	return (*y >= 0 && *y < heightImage && *x >= 0 && *x < widthImage);
}

 __device__ inline void CalculateBlockSizeXY(int widthImage, int heightImage, int& sizeX, int& sizeY)
 {
	 sizeX = widthImage - (blockDim.x) * (blockIdx.x);
	 sizeY = heightImage - (blockDim.y) * (blockIdx.y);
 }

 /** Devide 2 elements and rounf the result upwards.
 */
 template<class T> __device__  inline int DevideAndCeil(T a, T b)
{
	//return (T)((((float)(a)) / ((float)(b))) + 0.9999f);
	return (int)(((float)(a)) / ((float)(b)) + 0.9f);
 }

template<class T> __device__  inline T LimitResult(float result, T white)
{
	return (result < (float)white ? (T)result : white);
}

template<class T>  __device__  inline T RoundAndLimitResult(float result, T white)
{
	result = round(result);
	return (result < (float)white ? (T)result : white);
}

__device__ inline int Flip(bool doFlip, int height, int y)
{
	return ((doFlip) ? ((height - 1) - y) : (y));
}

__device__  inline int DecodeThreadIndex()
{
	return (threadIdx.z) + (blockDim.z) * ((threadIdx.x) + (threadIdx.y)*(blockDim.x)); 
}

__device__ inline float CalcualteDistance(float2 coord, float2 coord1)
{
	return sqrtf(SQR((coord.x - coord1.x)) + SQR((coord.y - coord1.y)));
}

__device__ inline float CalcualteDistance2D(float2 coord, float x0, float y0)
{
	return sqrtf(SQR((coord.x - x0)) + SQR((coord.y - y0)));
}