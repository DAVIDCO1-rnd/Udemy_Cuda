inline bool DecodeYXC_cpu(int* y, int* x, int* c, int widthImage, int heightImage, 
	int thread_Idx_x, int thread_Idx_y, int thread_Idx_z, int block_Idx_x, int block_Idx_y, int block_Dim_x, int block_Dim_y)
{
	*y = (thread_Idx_y) + (block_Dim_y)*(block_Idx_y);
	*x = (thread_Idx_x) + (block_Dim_x)*(block_Idx_x);
	*c = (thread_Idx_z);
	
	return (*y >= 0 && *y < heightImage && *x >= 0 && *x < widthImage);
}

inline int PixelOffset1D_cpu(int x, int channel, int pixelSize, int channelSize)
{
	return  x * pixelSize + channel * channelSize;
}

inline int PixelOffset_cpu(int y, int x, int channel, int stride, int pixelSize, int channelSize)
{
	return y * stride + PixelOffset1D_cpu(x, channel, pixelSize, channelSize);
}

template<class T> inline T* Pixel_cpu(void* buffer, int offset)
{
	return (T*)((unsigned char*)buffer + offset);
}