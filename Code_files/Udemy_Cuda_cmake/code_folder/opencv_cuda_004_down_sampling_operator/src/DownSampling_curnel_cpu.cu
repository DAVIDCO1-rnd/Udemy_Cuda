#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif

#pragma once

#include "Utils_device_cpu.cu"
#include "cutil_math.h"
#include "fake_cuda_in_cpu.cuh"

#define LIMITWEIGHT(x) (x) //( ((x)>=0.0f) ? ( ((x)<=1.0f) ? (x) : 0.0f ) : 0.0f )

#define DOWN_SAMPLING_EPSILON 1E-6f


// Support Downsampling for segmentation

template<class T> void DownSampleTopLeftKernel_cpu(
	unsigned char* inputData, unsigned char* outputData,
	int sourceWidth, int sourceHeight, int strideSource,
	int destWidth, int destHeight, int strideDest,
	float horizontalScale, float verticalScale,
	int white, int pixelSize, int channelSize,
	int blockSizeX, int blockSizeY, int blockSizeZ,
	int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int thread_Idx_z = 0; thread_Idx_z < blockSizeZ; thread_Idx_z++)
			{
				for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
				{
					for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
					{
						int destY = 0;
						int destX = 0;
						int channel = 0;
						if (!DecodeYXC_cpu(&destY, &destX, &channel, destWidth, destHeight, thread_Idx_x, thread_Idx_y, thread_Idx_z, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
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
						int indexSrc = PixelOffset_cpu(y, x, channel, strideSource, pixelSize, channelSize);
						float SegmentationPixelIDTarget = (*(Pixel_cpu<T>(inputData, indexSrc)));

						int indexDst = PixelOffset_cpu(destY, destX, channel, strideDest, pixelSize, channelSize);
						*(Pixel_cpu<T>(outputData, indexDst)) = RoundAndLimitResult_cpu<T>(SegmentationPixelIDTarget, white);
					}
				}
			}
		}
	}
}

template<class T> void DownSampleTopLeftKernelFloat_cpu(
	unsigned char* inputData, unsigned char* outputData,
	int sourceWidth, int sourceHeight, int strideSource,
	int destWidth, int destHeight, int strideDest,
	float horizontalScale, float verticalScale,
	int pixelSize, int channelSize,
	int blockSizeX, int blockSizeY, int blockSizeZ,
	int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int thread_Idx_z = 0; thread_Idx_z < blockSizeZ; thread_Idx_z++)
			{
				for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
				{
					for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
					{
						int destY = 0;
						int destX = 0;
						int channel = 0;
						if (!DecodeYXC_cpu(&destY, &destX, &channel, destWidth, destHeight, thread_Idx_x, thread_Idx_y, thread_Idx_z, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
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
						int indexSrc = PixelOffset_cpu(y, x, channel, strideSource, pixelSize, channelSize);
						float SegmentationPixelIDTarget = (*(Pixel_cpu<T>(inputData, indexSrc)));

						int indexDst = PixelOffset_cpu(destY, destX, channel, strideDest, pixelSize, channelSize);
						*(Pixel_cpu<T>(outputData, indexDst)) = (T)SegmentationPixelIDTarget;
					}
				}
			}
		}
	}
}


//-----------------------------------------------------------------------------------


template<class T> void DownSampleKernel_cpu(
				unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource,
				int destWidth, int destHeight, int strideDest,
				float horizontalScale, float verticalScale,
 				int white, int pixelSize, int channelSize,
				int blockSizeX, int blockSizeY, int blockSizeZ,
				int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int thread_Idx_z = 0; thread_Idx_z < blockSizeZ; thread_Idx_z++)
			{
				for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
				{
					for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
					{
						int destY = 0;
						int destX = 0;
						int channel = 0;
						if (!DecodeYXC_cpu(&destY, &destX, &channel, destWidth, destHeight, thread_Idx_x, thread_Idx_y, thread_Idx_z, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
							return;

						if (destX == ((int)(0.25 * destWidth)) && destY == ((int)(0.35 * destHeight)))
						{
							int david = 5;
						}

						// Calculate source's center
						// Calculate target's center
						// Centralize target coordinates
						// Calculate source pixel center in centralized coordinates
						// De-Centralize coordinates

						float dest_centerX = (destWidth - 1) * 0.5f;
						float dest_centerY = (destHeight - 1) * 0.5f;

						float diff_dest_center_x = destX - dest_centerX;
						float diff_dest_center_y = destY - dest_centerY;

						float scaled_diff_dest_center_x = diff_dest_center_x * horizontalScale;
						float scaled_diff_dest_center_y = diff_dest_center_y * verticalScale;

						float source_centerX = (sourceWidth - 1) * 0.5f;
						float source_centerY = (sourceHeight - 1) * 0.5f;

						float sourceX = scaled_diff_dest_center_x + source_centerX;
						float sourceY = scaled_diff_dest_center_y + source_centerY;

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

								int indexSrc = PixelOffset_cpu(y, x, channel, strideSource, pixelSize, channelSize);
								T pixel_val = *(Pixel_cpu<T>(inputData, indexSrc));
								float val_to_add = pixel_val * wTop * wBottom * wLeft * wRight;
								sum += val_to_add;
							}
						}

						float average_pixel = sum / (horizontalScale * verticalScale);

						int indexDst = PixelOffset_cpu(destY, destX, channel, strideDest, pixelSize, channelSize);
						T rounded_val = RoundAndLimitResult_cpu<T>(average_pixel, white);
						*(Pixel_cpu<T>(outputData, indexDst)) = rounded_val;
					}
				}
			}
		}
	}
}

template<class T> void DownSampleKernelFloat_cpu(
				unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource,
				int destWidth, int destHeight, int strideDest,
				float horizontalScale, float verticalScale,
 				int pixelSize, int channelSize,
				int blockSizeX, int blockSizeY, int blockSizeZ,
				int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int thread_Idx_z = 0; thread_Idx_z < blockSizeZ; thread_Idx_z++)
			{
				for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
				{
					for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
					{
						int destY = 0;
						int destX = 0;
						int channel = 0;
						if (!DecodeYXC_cpu(&destY, &destX, &channel, destWidth, destHeight, thread_Idx_x, thread_Idx_y, thread_Idx_z, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
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

								int indexSrc = PixelOffset_cpu(y, x, channel, strideSource, pixelSize, channelSize);
								sum += (*(Pixel_cpu<T>(inputData, indexSrc))) * wTop * wBottom * wLeft * wRight;
							}
						}

						sum /= (horizontalScale * verticalScale);

						int indexDst = PixelOffset_cpu(destY, destX, channel, strideDest, pixelSize, channelSize);
						*(Pixel_cpu<T>(outputData, indexDst)) = (T)sum;
					}
				}
			}
		}
	}
}

template<int channels>
inline void addFraction_cpu(unsigned char* outputData, float4 value, float fraction, int indexPixel)
{
	if (fraction < DOWN_SAMPLING_EPSILON)
		return;

	atomicAdd_cpu(Pixel_cpu<int>(outputData, indexPixel), (int)rintf(fraction * value.x));
	if (channels > 1)
	{
		atomicAdd_cpu(Pixel_cpu<int>(outputData, indexPixel + GRAYFLOAT_SIZE), (int)rintf(fraction * value.y));
		atomicAdd_cpu(Pixel_cpu<int>(outputData, indexPixel + GRAYFLOAT_SIZE * 2), (int)rintf(fraction * value.z));
		if (channels > 3)
			atomicAdd_cpu(Pixel_cpu<int>(outputData, indexPixel + GRAYFLOAT_SIZE * 3), (int)rintf(fraction * value.w));
	}
}

template<class T, int channels, int channelSizeSource>
void FastDownSampleKernel_cpu(unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource, 
				 int strideDest,
				float horizontalScale, float verticalScale,
				float convertValue,
				int blockSizeX, int blockSizeY,
				int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
			{
				for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
				{
					int rowSrc = 0;
					int columnSrc = 0;
					if (!DecodeYX_cpu(&rowSrc, &columnSrc, sourceWidth, sourceHeight, thread_Idx_x, thread_Idx_y, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
						return;

					int indexSrc = PixelOffset_cpu(rowSrc, columnSrc, 0, strideSource, channelSizeSource * channels, channelSizeSource);

					float4 pixel = make_float4(0);

					pixel.x = (float)(*Pixel_cpu<T>(inputData, indexSrc)) * convertValue;

					if (channels > 1)
					{
						pixel.y = (float)(*Pixel_cpu<T>(inputData, indexSrc + 1 * channelSizeSource)) * convertValue;
						pixel.z = (float)(*Pixel_cpu<T>(inputData, indexSrc + 2 * channelSizeSource)) * convertValue;
						if (channels > 3)
							pixel.w = (float)(*Pixel_cpu<T>(inputData, indexSrc + 3 * channelSizeSource)) * convertValue;
					}

					syncthreads_cpu();

					int destWidth = ceil(horizontalScale * sourceWidth);
					int destHeight = ceil(verticalScale * sourceHeight);

					float rowDst = ((float)rowSrc - ((float)(sourceHeight)) * 0.5f) * verticalScale + ((float)(destHeight)) * 0.5f;
					if (rowDst < 0.0f)
						rowDst = 0.0f;
					int rowMinDst = floor(rowDst);
					int rowMaxDst = floor(rowDst + verticalScale);
					if (rowMaxDst > destHeight)
						rowMaxDst = destHeight;
					float rowDeltaDst = 1.0f;
					if (rowMinDst != rowMaxDst)
						rowDeltaDst = (rowMaxDst - rowDst) / verticalScale;


					float columnDst = ((float)columnSrc - ((float)(sourceWidth)) * 0.5f) * horizontalScale + ((float)(destWidth)) * 0.5f;
					if (columnDst < 0.0f)
						columnDst = 0.0f;
					int columnMinDst = floorf(columnDst);
					int columnMaxDst = floorf(columnDst + horizontalScale);
					if (columnMaxDst > destWidth)
						columnMaxDst = destWidth;
					float columnDeltaDst = 1.0f;
					if (columnMinDst != columnMaxDst)
						columnDeltaDst = ((float)(columnMaxDst - columnDst)) / horizontalScale;


					// Upper Left
					int indexDest = PixelOffset_cpu(rowMinDst, columnMinDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
					addFraction_cpu<channels>(outputData, pixel, columnDeltaDst * rowDeltaDst, indexDest);

					// Upper Right
					indexDest = PixelOffset_cpu(rowMaxDst, columnMinDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
					addFraction_cpu<channels>(outputData, pixel, columnDeltaDst * (1.0f - rowDeltaDst), indexDest);

					// Lowwer Left
					indexDest = PixelOffset_cpu(rowMinDst, columnMaxDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
					addFraction_cpu<channels>(outputData, pixel, (1.0f - columnDeltaDst) * rowDeltaDst, indexDest);

					// Lowwer Right
					indexDest = PixelOffset_cpu(rowMaxDst, columnMaxDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
					addFraction_cpu<channels>(outputData, pixel, (1.0f - columnDeltaDst) * (1.0f - rowDeltaDst), indexDest);
				}
			}
		}
	}
}


template<class T, int channels, int channelSizeSource>
void FastDownSampleVerticalKernel_cpu(unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource,
				int strideDest,
				float verticalScale, float convertValue,
				int blockSizeX, int blockSizeY,
				int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
			{
				for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
				{
					int rowSrc = 0;
					int columnSrc = 0;
					if (!DecodeYX_cpu(&rowSrc, &columnSrc, sourceWidth, sourceHeight, thread_Idx_x, thread_Idx_y, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
						return;

					int indexSrc = PixelOffset_cpu(rowSrc, columnSrc, 0, strideSource, channelSizeSource * channels, channelSizeSource);

					float4 pixel = make_float4(0);

					pixel.x = (float)(*Pixel_cpu<T>(inputData, indexSrc)) * convertValue;

					if (channels > 1)
					{
						pixel.y = (float)(*Pixel_cpu<T>(inputData, indexSrc + 1 * channelSizeSource)) * convertValue;
						pixel.z = (float)(*Pixel_cpu<T>(inputData, indexSrc + 2 * channelSizeSource)) * convertValue;
						if (channels > 3)
							pixel.w = (float)(*Pixel_cpu<T>(inputData, indexSrc + 3 * channelSizeSource)) * convertValue;
					}

					syncthreads_cpu();

					int destHeight = ceil(verticalScale * sourceHeight);

					float rowDst = ((float)rowSrc - ((float)(sourceHeight)) * 0.5f) * verticalScale + ((float)(destHeight)) * 0.5f;
					if (rowDst < 0.0f)
						rowDst = 0.0f;
					int rowMinDst = floor(rowDst);
					int rowMaxDst = floor(rowDst + verticalScale);
					float rowDeltaDst = 1.0f;
					if (rowMinDst != rowMaxDst)
						rowDeltaDst = ((float)(rowMaxDst - rowDst)) / verticalScale;


					// Upper Left
					int indexDest = PixelOffset_cpu(rowMinDst, columnSrc, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
					addFraction_cpu<channels>(outputData, pixel, rowDeltaDst, indexDest);

					// Upper Right
					indexDest = PixelOffset_cpu(rowMaxDst, columnSrc, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
					addFraction_cpu<channels>(outputData, pixel, (1.0f - rowDeltaDst), indexDest);
				}
			}
		}
	}
}

template<class T, int channels, int channelSizeSource>
void FastDownSampleHorizontalKernel_cpu(unsigned char* inputData, unsigned char* outputData,
				int sourceWidth, int sourceHeight, int strideSource,
				int strideDest, float horizontalScale, float convertValue,
				int blockSizeX, int blockSizeY,
				int gridSizeX, int gridSizeY)
{
	int block_Dim_x = blockSizeX;
	int block_Dim_y = blockSizeY;

	for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
	{
		for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
		{
			for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
			{
				for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
				{
					int rowSrc = 0;
					int columnSrc = 0;
					if (!DecodeYX_cpu(&rowSrc, &columnSrc, sourceWidth, sourceHeight, thread_Idx_x, thread_Idx_y, block_Idx_x, block_Idx_y, block_Dim_x, block_Dim_y))
						return;

					int indexSrc = PixelOffset_cpu(rowSrc, columnSrc, 0, strideSource, channelSizeSource * channels, channelSizeSource);

					float4 pixel = make_float4(0);

					pixel.x = (float)(*Pixel_cpu<T>(inputData, indexSrc)) * convertValue;

					if (channels > 1)
					{
						pixel.y = (float)(*Pixel_cpu<T>(inputData, indexSrc + 1 * channelSizeSource)) * convertValue;
						pixel.z = (float)(*Pixel_cpu<T>(inputData, indexSrc + 2 * channelSizeSource)) * convertValue;
						if (channels > 3)
							pixel.w = (float)(*Pixel_cpu<T>(inputData, indexSrc + 3 * channelSizeSource)) * convertValue;
					}

					syncthreads_cpu();

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
					int indexDest = PixelOffset_cpu(rowSrc, columnMinDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
					addFraction_cpu<channels>(outputData, pixel, columnDeltaDst, indexDest);

					// Lowwer Left
					indexDest = PixelOffset_cpu(rowSrc, columnMaxDst, 0, strideDest, GRAYFLOAT_SIZE * channels, GRAYFLOAT_SIZE);
					addFraction_cpu<channels>(outputData, pixel, (1.0f - columnDeltaDst), indexDest);
				}
			}
		}
	}
}
