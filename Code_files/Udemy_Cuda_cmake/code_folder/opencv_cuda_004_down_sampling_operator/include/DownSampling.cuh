#pragma once


extern "C" __declspec(dllexport)
__wchar_t* DownSampleTopLeft(
		void* deviceInputBuffer, void* deviceOutputBuffer,
		int widthSourceImage, int heightSourceImage, int strideSourceImage,
		int widthDestImage, int heightDestImage, int strideDestImage,
		double horizontalScale, double verticalScale,
		int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,
		int blockSizeX, int blockSizeY, int blockSizeZ, int gridSizeX, int gridSizeY);



extern "C" __declspec(dllexport)
__wchar_t* DownSampleTopLeft_Parallel(
		void* deviceInputBuffer, void* deviceOutputBuffer,
		int widthSourceImage, int heightSourceImage, int strideSourceImage,
		int widthDestImage, int heightDestImage, int strideDestImage,
		double horizontalScale, double verticalScale,
		int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,
		int blockSizeX, int blockSizeY, int blockSizeZ, int gridSizeX, int gridSizeY, void* stream);


extern "C" __declspec( dllexport )
__wchar_t* DownSample(
		 void* deviceInputBuffer, void* deviceOutputBuffer,
		 int widthSourceImage, int heightSourceImage, int strideSourceImage,
		 int widthDestImage, int heightDestImage, int strideDestImage,
		 double horizontalScale, double verticalScale,
		 int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,			
		 int blockSizeX, int blockSizeY, int blockSizeZ, int gridSizeX, int gridSizeY);

extern "C" __declspec( dllexport )
__wchar_t* DownSample_Parallel(
		 void* deviceInputBuffer, void* deviceOutputBuffer,
		 int widthSourceImage, int heightSourceImage, int strideSourceImage,
		 int widthDestImage, int heightDestImage, int strideDestImage,
		 double horizontalScale, double verticalScale,
		 int subPixelType, int maxValue, int alphaChannelNumber, int pixelSize, int channelSize,			
		 int blockSizeX, int blockSizeY, int blockSizeZ, int gridSizeX, int gridSizeY, void* stream);

extern "C" __declspec( dllexport )
__wchar_t* FastDownSample(void* deviceInputBuffer, void* deviceIntegerOutputBuffer, 
								   void* deviceOutputBuffer, double maxValue, double convertValue, 
								   int numberOfChannels, int subPixelType, 
								   int widthSourceImage, int heightSourceImage, int strideSourceImage,
								   int widthDestImage, int heightDestImage, int strideIntegerDestImage, int strideDestImage,
								   double horizontalScale, double verticalScale,
								   int blockSizeSrcX, int blockSizeSrcY, int gridSizeSrcX, int gridSizeSrcY, 		
								   int blockSizeDstX, int blockSizeDstY, int gridSizeDstX, int gridSizeDstY);


extern "C" __declspec( dllexport )
__wchar_t* FastDownSample_Parallel(void* deviceInputBuffer, void* deviceIntegerOutputBuffer, 
								   void* deviceOutputBuffer, double maxValue, double convertValue, 
								   int numberOfChannels, int subPixelType, 
								   int widthSourceImage, int heightSourceImage, int strideSourceImage,
								   int widthDestImage, int heightDestImage, int strideIntegerDestImage, int strideDestImage,
								   double horizontalScale, double verticalScale,
								   int blockSizeSrcX, int blockSizeSrcY, int gridSizeSrcX, int gridSizeSrcY, 
								   int blockSizeDstX, int blockSizeDstY, int gridSizeDstX, int gridSizeDstY,
								   void* stream);
								   