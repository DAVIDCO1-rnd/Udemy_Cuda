#pragma once

/**
 * Inverses buffer of source image and save result in buffer of destination image
 * @param inputBuffer buffer of source image
 * @param outputBuffer buffer of destination image
 */
extern "C" __declspec( dllexport )
	__wchar_t* Inverse (void* deviceInputBuffer, void* deviceOutputBuffer,
				   int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,
				   int widthImage, int heightImage, int strideSourceImage, int strideResultImage,
				   int blockSizeX,int blockSizeY,int blockSizeZ,
				   int gridSizeX, int gridSizeY);
