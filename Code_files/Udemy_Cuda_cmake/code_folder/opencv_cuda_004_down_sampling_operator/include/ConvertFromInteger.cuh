#pragma once

template< class T>
__wchar_t* RunConvertFromIntegerKernel(void* deviceInputData, void* deviceOutputBuffer,
									 int numberOfChannels,
									 double saturation, double convertValue,
									  int widthImage, int heightImage, int strideSrcImage, int strideDstImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream);

__wchar_t* RunConvertFromIntegerToFloatKernel(void* deviceInputData, void* deviceOutputBuffer,
									 int numberOfChannels,
									 double saturation, double convertValue,
									  int widthImage, int heightImage, int strideSrcImage, int strideDstImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream);


__wchar_t* ConvertFromIntegerToImage_Internal(void* deviceInputData, void* deviceOutputBuffer,
									 int outputSubPixelType, int numberOfChannels,
									 double saturation, double convertValue,
									  int widthImage, int heightImage, int strideSrcImage, int strideDstImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream);


__wchar_t* ConvertFromIntegeToFloat__Internal(void* deviceData, 
									 int numberOfChannels, double convertValue,
									  int widthImage, int heightImage, int strideImage,
									  int blockSizeX,int blockSizeY,
									  int gridSizeX, int gridSizeY, void* stream);