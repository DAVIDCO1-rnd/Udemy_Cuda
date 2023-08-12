#pragma once

/**
 * Inverses buffer of source image and save result in buffer of destination image
 * @param inputBuffer buffer of source image
 * @param outputBuffer buffer of destination image
 */
extern "C" __declspec( dllexport )
	__wchar_t * rotate_90(	void* deviceInputBuffer, void* deviceOutputBuffer, int subPixelType,
							int widthImage, int heightImage, int is_clockwise,
							int blockSizeX, int blockSizeY, int blockSizeZ,
							int gridSizeX, int gridSizeY);
