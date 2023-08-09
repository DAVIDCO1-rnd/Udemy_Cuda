#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#pragma once

#include "Utils_device.cu"
#include "cutil_math.h"

#define LIMITWEIGHT(x) (x) //( ((x)>=0.0f) ? ( ((x)<=1.0f) ? (x) : 0.0f ) : 0.0f )

#define DOWN_SAMPLING_EPSILON 1E-6f


// Support Downsampling for segmentation

template<class T> __global__ void DownSampleTopLeftKernel(
	unsigned char* inputData, unsigned char* outputData,
	int sourceWidth, int sourceHeight, int strideSource,
	int destWidth, int destHeight, int strideDest,
	float horizontalScale, float verticalScale,
	int white, int pixelSize, int channelSize)
{
	int destY = 0;
	int destX = 0;
	int channel = 0;
	if (!DecodeYXC(&destY, &destX, &channel, destWidth, destHeight))
		return;

	// Calculate source's center
	// Calculate target's center
	// Centralize target coordinates
	// Calculate source pixel center in centralized coordinates
	// De-Centralize coordinates
	float sourceX = (destX - (destWidth - 1) * 0.5f) * horizontalScale + (sourceWidth - 1) * 0.5f;
	float sourceY = (destY - (destHeight - 1) * 0.5f) * verticalScale + (sourceHeight - 1) * 0.5f;

	// Calculate source range that is averaged to target range
	float xMinRange = sourceX - 0.5f * horizontalScale;
	float xMaxRange = sourceX + 0.5f * horizontalScale;
	float yMinRange = sourceY - 0.5f * verticalScale;
	float yMaxRange = sourceY + 0.5f * verticalScale;

	int xMinLimit = MAX(0, floor(xMinRange + 0.5f));
	int xMaxLimit = MIN(sourceWidth - 1, ceil(xMaxRange - 0.5f));
	int yMinLimit = MAX(0, floor(yMinRange + 0.5f));
	int yMaxLimit = MIN(sourceHeight - 1, ceil(yMaxRange - 0.5f));
		
	int x = xMinLimit;
	int y = yMinLimit;
	int indexSrc = PixelOffset(y, x, channel, strideSource, pixelSize, channelSize);
	float SegmentationPixelIDTarget = (*(Pixel<T>(inputData, indexSrc)));	

	int indexDst = PixelOffset(destY, destX, channel, strideDest, pixelSize, channelSize);
	*(Pixel<T>(outputData, indexDst)) = RoundAndLimitResult<T>(SegmentationPixelIDTarget, white);
}

template<class T> __global__ void DownSampleTopLeftKernelFloat(
	unsigned char* inputData, unsigned char* outputData,
	int sourceWidth, int sourceHeight, int strideSource,
	int destWidth, int destHeight, int strideDest,
	float horizontalScale, float verticalScale,
	int pixelSize, int channelSize)
{
	int destY = 0;
	int destX = 0;
	int channel = 0;
	if (!DecodeYXC(&destY, &destX, &channel, destWidth, destHeight))
		return;

	// Calculate source's center
	// Calculate target's center
	// Centralize target coordinates
	// Calculate source pixel center in centralized coordinates
	// De-Centralize coordinates
	float sourceX = (destX - (destWidth - 1) * 0.5f) * horizontalScale + (sourceWidth - 1) * 0.5f;
	float sourceY = (destY - (destHeight - 1) * 0.5f) * verticalScale + (sourceHeight - 1) * 0.5f;

	// Calculate source range that is averaged to target range
	float xMinRange = sourceX - 0.5f * horizontalScale;
	float xMaxRange = sourceX + 0.5f * horizontalScale;
	float yMinRange = sourceY - 0.5f * verticalScale;
	float yMaxRange = sourceY + 0.5f * verticalScale;

	int xMinLimit = MAX(0, floor(xMinRange + 0.5f));
	int xMaxLimit = MIN(sourceWidth - 1, ceil(xMaxRange - 0.5f));
	int yMinLimit = MAX(0, floor(yMinRange + 0.5f));
	int yMaxLimit = MIN(sourceHeight - 1, ceil(yMaxRange - 0.5f));


	int x = xMinLimit;
	int y = yMinLimit;
	int indexSrc = PixelOffset(y, x, channel, strideSource, pixelSize, channelSize);
	float SegmentationPixelIDTarget = (*(Pixel<T>(inputData, indexSrc)));

	int indexDst = PixelOffset(destY, destX, channel, strideDest, pixelSize, channelSize);
	*(Pixel<T>(outputData, indexDst)) = (T)SegmentationPixelIDTarget;
}


//-----------------------------------------------------------------------------------


template<class T> __global__ void DownSampleKernel(
				unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource,
				int destWidth, int destHeight, int strideDest,
				float horizontalScale, float verticalScale,
 				int white, int pixelSize, int channelSize)
{
	int destY = 0;
	int destX = 0;
	int channel = 0;
	if (!DecodeYXC(&destY, &destX, &channel, destWidth, destHeight))
		return;	

	// Calculate source's center
	// Calculate target's center
	// Centralize target coordinates
	// Calculate source pixel center in centralized coordinates
	// De-Centralize coordinates
	float sourceX = (destX - (destWidth - 1) * 0.5f) * horizontalScale + (sourceWidth - 1) * 0.5f;
	float sourceY = (destY - (destHeight - 1) * 0.5f) * verticalScale + (sourceHeight - 1) * 0.5f;

	// Calculate source range that is averaged to target range
	float xMinRange = sourceX - 0.5f * horizontalScale;
	float xMaxRange = sourceX + 0.5f * horizontalScale;
	float yMinRange = sourceY - 0.5f * verticalScale;
	float yMaxRange = sourceY + 0.5f * verticalScale;

	int xMinLimit = MAX(0, floor(xMinRange + 0.5f));
	int xMaxLimit = MIN(sourceWidth - 1, ceil(xMaxRange - 0.5f));
	int yMinLimit = MAX(0, floor(yMinRange + 0.5f));
	int yMaxLimit = MIN(sourceHeight - 1, ceil(yMaxRange - 0.5f));
	
	float sum = 0.0f;
	float wTop, wBottom, wLeft, wRight;

	for (int y = yMinLimit; y <= yMaxLimit; y++)
	{
		wTop = MIN((y + 0.5f) - yMinRange, 1);
        wBottom = MIN(yMaxRange - (y - 0.5f), 1);

		for (int x = xMinLimit; x <= xMaxLimit; x++)
		{
			wLeft = MIN((x + 0.5f) - xMinRange, 1);
			wRight = MIN(xMaxRange - (x - 0.5f), 1);

			int indexSrc = PixelOffset(y, x, channel, strideSource, pixelSize, channelSize);
			sum += (*(Pixel<T>(inputData, indexSrc))) * wTop * wBottom * wLeft * wRight;
		}
	}

	sum /= (horizontalScale * verticalScale);

	int indexDst = PixelOffset(destY, destX, channel, strideDest, pixelSize, channelSize);
	*(Pixel<T>(outputData, indexDst)) = RoundAndLimitResult<T>(sum, white); 
}

template<class T> __global__ void DownSampleKernelFloat(
				unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource,
				int destWidth, int destHeight, int strideDest,
				float horizontalScale, float verticalScale,
 				int pixelSize, int channelSize)
{
	int destY = 0;
	int destX = 0;
	int channel = 0;
	if (!DecodeYXC(&destY, &destX, &channel, destWidth, destHeight))
		return;	

	// Calculate source's center
	// Calculate target's center
	// Centralize target coordinates
	// Calculate source pixel center in centralized coordinates
	// De-Centralize coordinates
	float sourceX = (destX - (destWidth - 1) * 0.5f) * horizontalScale + (sourceWidth - 1) * 0.5f;
	float sourceY = (destY - (destHeight - 1) * 0.5f) * verticalScale + (sourceHeight - 1) * 0.5f;

	// Calculate source range that is averaged to target range
	float xMinRange = sourceX - 0.5f * horizontalScale;
	float xMaxRange = sourceX + 0.5f * horizontalScale;
	float yMinRange = sourceY - 0.5f * verticalScale;
	float yMaxRange = sourceY + 0.5f * verticalScale;

	int xMinLimit = MAX(0, floor(xMinRange + 0.5f));
	int xMaxLimit = MIN(sourceWidth - 1, ceil(xMaxRange - 0.5f));
	int yMinLimit = MAX(0, floor(yMinRange + 0.5f));
	int yMaxLimit = MIN(sourceHeight - 1, ceil(yMaxRange - 0.5f));
	
	float sum = 0.0f;
	float wTop, wBottom, wLeft, wRight;

	for (int y = yMinLimit; y <= yMaxLimit; y++)
	{
		wTop = MIN((y + 0.5f) - yMinRange, 1);
        wBottom = MIN(yMaxRange - (y - 0.5f), 1);

		for (int x = xMinLimit; x <= xMaxLimit; x++)
		{
			wLeft = MIN((x + 0.5f) - xMinRange, 1);
			wRight = MIN(xMaxRange - (x - 0.5f), 1);

			int indexSrc = PixelOffset(y, x, channel, strideSource, pixelSize, channelSize);
			sum += (*(Pixel<T>(inputData, indexSrc))) * wTop * wBottom * wLeft * wRight;
		}
	}

	sum /= (horizontalScale * verticalScale);

	int indexDst = PixelOffset(destY, destX, channel, strideDest, pixelSize, channelSize);
	*(Pixel<T>(outputData, indexDst)) = (T)sum; 
}

template<int channels>
__device__ inline void addFraction(unsigned char* outputData, float4 value, float fraction, int indexPixel)
{
	if (fraction < DOWN_SAMPLING_EPSILON)
		return;

	atomicAdd(Pixel<int>(outputData, indexPixel), (int)rintf(fraction * value.x));
	if (channels > 1)
	{
		atomicAdd(Pixel<int>(outputData, indexPixel + GRAYFLOAT_SIZE), (int)rintf(fraction * value.y));
		atomicAdd(Pixel<int>(outputData, indexPixel + GRAYFLOAT_SIZE * 2), (int)rintf(fraction * value.z));
		if (channels > 3)
			atomicAdd(Pixel<int>(outputData, indexPixel + GRAYFLOAT_SIZE * 3), (int)rintf(fraction * value.w));
	}
}

template<class T, int channels, int channelSizeSource>
__global__ void FastDownSampleKernel(unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource, 
				 int strideDest,
				float horizontalScale, float verticalScale,
				float convertValue)
{
	int rowSrc = 0;
	int columnSrc = 0;
	if (!DecodeYX(&rowSrc, &columnSrc, sourceWidth, sourceHeight))
		return;	

	int indexSrc = PixelOffset(rowSrc, columnSrc, 0, strideSource, channelSizeSource * channels, channelSizeSource);

	float4 pixel = make_float4(0);
	
	pixel.x = (float)(*Pixel<T>(inputData, indexSrc)) * convertValue;

	if (channels > 1)
	{
		pixel.y =(float)(*Pixel<T>(inputData, indexSrc + 1 * channelSizeSource)) * convertValue;
		pixel.z =(float)(*Pixel<T>(inputData, indexSrc + 2 * channelSizeSource))* convertValue;
		if (channels > 3)
			pixel.w =(float)(*Pixel<T>(inputData, indexSrc + 3 * channelSizeSource)) * convertValue;
	}

	__syncthreads();

	int destWidth = ceil(horizontalScale * sourceWidth);
	int destHeight = ceil(verticalScale * sourceHeight);

	float rowDst =  ((float)rowSrc - ((float)(sourceHeight)) * 0.5f) * verticalScale + ((float)(destHeight)) * 0.5f;
	if (rowDst < 0.0f) 
		rowDst = 0.0f;
	int rowMinDst = floor(rowDst);
	int rowMaxDst = floor(rowDst + verticalScale);
	if(rowMaxDst > destHeight)
		rowMaxDst = destHeight;
	float rowDeltaDst = 1.0f;
	if (rowMinDst != rowMaxDst)
		rowDeltaDst = (rowMaxDst - rowDst) / verticalScale;


	float columnDst = ((float)columnSrc - ((float)(sourceWidth)) * 0.5f) * horizontalScale + ((float)(destWidth)) * 0.5f;
	if (columnDst < 0.0f) 
		columnDst = 0.0f;
	int columnMinDst = floorf(columnDst);
	int columnMaxDst = floorf(columnDst + horizontalScale);
	if(columnMaxDst > destWidth)
		columnMaxDst = destWidth;
	float columnDeltaDst = 1.0f;
	if (columnMinDst != columnMaxDst)
		columnDeltaDst = ((float)(columnMaxDst - columnDst)) / horizontalScale;


	// Upper Left
	int indexDest =  PixelOffset(rowMinDst, columnMinDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
	addFraction<channels>(outputData, pixel,columnDeltaDst * rowDeltaDst , indexDest);

	// Upper Right
	indexDest =  PixelOffset(rowMaxDst, columnMinDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
	addFraction<channels>(outputData, pixel, columnDeltaDst * (1.0f - rowDeltaDst) , indexDest);

	// Lowwer Left
	indexDest =  PixelOffset(rowMinDst, columnMaxDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
	addFraction<channels>(outputData, pixel, (1.0f - columnDeltaDst) * rowDeltaDst , indexDest);

	// Lowwer Right
	indexDest =  PixelOffset(rowMaxDst, columnMaxDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
	addFraction<channels>(outputData, pixel, (1.0f - columnDeltaDst) * (1.0f - rowDeltaDst) , indexDest);
}


template<class T, int channels, int channelSizeSource>
__global__ void FastDownSampleVerticalKernel(unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource,
				int strideDest,
				float verticalScale, float convertValue)
{
	int rowSrc = 0;
	int columnSrc = 0;
	if (!DecodeYX(&rowSrc, &columnSrc, sourceWidth, sourceHeight))
		return;	

	int indexSrc = PixelOffset(rowSrc, columnSrc, 0, strideSource, channelSizeSource * channels, channelSizeSource);

	float4 pixel = make_float4(0);
	
	pixel.x = (float)(*Pixel<T>(inputData, indexSrc)) * convertValue;

	if (channels > 1)
	{
		pixel.y =(float)(*Pixel<T>(inputData, indexSrc + 1 * channelSizeSource)) * convertValue;
		pixel.z =(float)(*Pixel<T>(inputData, indexSrc + 2 * channelSizeSource))* convertValue;
		if (channels > 3)
			pixel.w =(float)(*Pixel<T>(inputData, indexSrc + 3 * channelSizeSource)) * convertValue;
	}

	__syncthreads();

	int destHeight = ceil(verticalScale * sourceHeight);

	float rowDst =  ((float)rowSrc - ((float)(sourceHeight)) * 0.5f) * verticalScale + ((float)(destHeight)) * 0.5f;
	if (rowDst < 0.0f) 
		rowDst = 0.0f;
	int rowMinDst = floor(rowDst);
	int rowMaxDst = floor(rowDst + verticalScale);
	float rowDeltaDst = 1.0f;
	if (rowMinDst != rowMaxDst)
		rowDeltaDst = ((float)(rowMaxDst - rowDst)) / verticalScale;


	// Upper Left
	int indexDest =  PixelOffset(rowMinDst, columnSrc, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
	addFraction<channels>(outputData, pixel, rowDeltaDst , indexDest);

	// Upper Right
	indexDest =  PixelOffset(rowMaxDst, columnSrc, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
	addFraction<channels>(outputData, pixel,  (1.0f - rowDeltaDst) , indexDest);

}

template<class T, int channels, int channelSizeSource>
__global__ void FastDownSampleHorizontalKernel(unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource,
				int strideDest, float horizontalScale, float convertValue)
{
	int rowSrc = 0;
	int columnSrc = 0;
	if (!DecodeYX(&rowSrc, &columnSrc, sourceWidth, sourceHeight))
		return;	

	int indexSrc = PixelOffset(rowSrc, columnSrc, 0, strideSource, channelSizeSource * channels, channelSizeSource);

	float4 pixel = make_float4(0);
	
	pixel.x = (float)(*Pixel<T>(inputData, indexSrc)) * convertValue;

	if (channels > 1)
	{
		pixel.y =(float)(*Pixel<T>(inputData, indexSrc + 1 * channelSizeSource)) * convertValue;
		pixel.z =(float)(*Pixel<T>(inputData, indexSrc + 2 * channelSizeSource))* convertValue;
		if (channels > 3)
			pixel.w =(float)(*Pixel<T>(inputData, indexSrc + 3 * channelSizeSource)) * convertValue;
	}

	__syncthreads();

	int destWidth = ceil(horizontalScale * sourceWidth);


	float columnDst = ((float)columnSrc - ((float)(sourceWidth)) * 0.5f) * horizontalScale + ((float)(destWidth)) * 0.5f;
	if (columnDst < 0.0f) 
		columnDst = 0.0f;
	int columnMinDst = floor(columnDst);
	int columnMaxDst = floor(columnDst + horizontalScale);
	float columnDeltaDst = 1.0f;
	if (columnMinDst != columnMaxDst)
		columnDeltaDst = ((float)(columnMaxDst - columnDst)) / horizontalScale;


	// Upper Left
	int indexDest =  PixelOffset(rowSrc, columnMinDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
	addFraction<channels>(outputData, pixel,columnDeltaDst  , indexDest);

	// Lowwer Left
	indexDest =  PixelOffset(rowSrc, columnMaxDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
	addFraction<channels>(outputData, pixel, (1.0f - columnDeltaDst) , indexDest);
}
