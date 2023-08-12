
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

inline bool DecodeYXC_cpu(int* y, int* x, int* c, int widthImage, int heightImage, 
	int thread_Idx_x, int thread_Idx_y, int thread_Idx_z, int block_Idx_x, int block_Idx_y, int block_Dim_x, int block_Dim_y)
{
	*y = (thread_Idx_y) + (block_Dim_y)*(block_Idx_y);
	*x = (thread_Idx_x) + (block_Dim_x)*(block_Idx_x);
	*c = (thread_Idx_z);
	
	return (*y >= 0 && *y < heightImage && *x >= 0 && *x < widthImage);
}

inline bool DecodeYX_cpu(int* y, int* x, int widthImage, int heightImage,
	int thread_Idx_x, int thread_Idx_y, int block_Idx_x, int block_Idx_y, int block_Dim_x, int block_Dim_y)
{
	*y = (thread_Idx_y) + (block_Dim_y) * (block_Idx_y);
	*x = (thread_Idx_x) + (block_Dim_x) * (block_Idx_x);

	return (*y >= 0 && *y < heightImage&&* x >= 0 && *x < widthImage);
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

/** Devide 2 elements and rounf the result upwards.
*/
template<class T> inline int DevideAndCeil_cpu(T a, T b)
{
	//return (T)((((float)(a)) / ((float)(b))) + 0.9999f);
	return (int)(((float)(a)) / ((float)(b)) + 0.9f);
}

template<class T> inline T LimitResult_cpu(float result, T white)
{
	return (result < (float)white ? (T)result : white);
}

template<class T>  inline T RoundAndLimitResult_cpu(float result, T white)
{
	result = round(result);
	return (result < (float)white ? (T)result : white);
}

inline int Flip_cpu(bool doFlip, int height, int y)
{
	return ((doFlip) ? ((height - 1) - y) : (y));
}

inline int DecodeThreadIndex_cpu(int thread_Idx_x, int thread_Idx_y, int thread_Idx_z, int block_Dim_x, int block_Dim_z)
{
	return (thread_Idx_z) + (block_Dim_z) * ((thread_Idx_x) + (thread_Idx_y) * (block_Dim_x));
}

inline float CalcualteDistance_cpu(float2 coord, float2 coord1)
{
	return sqrtf(SQR((coord.x - coord1.x)) + SQR((coord.y - coord1.y)));
}

inline float CalcualteDistance2D_cpu(float2 coord, float x0, float y0)
{
	return sqrtf(SQR((coord.x - x0)) + SQR((coord.y - y0)));
}