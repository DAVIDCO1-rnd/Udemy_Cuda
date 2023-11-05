#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <Windows.h>

std::vector <std::string> gToPrint;

void PrintTimeStamp(const char* prefix)
{
	char tempBuff[2000];
	SYSTEMTIME ttt;
	GetSystemTime(&ttt);
	long timeMS = (ttt.wSecond * 1000) + ttt.wMilliseconds;
	sprintf(tempBuff, "-D- Time in %s is %llx", prefix, timeMS);
	gToPrint.push_back(tempBuff);
}

void PrintAll()
{
	for (unsigned int it = 0; it < gToPrint.size(); ++it)
	{
		printf("%s\n", gToPrint[it].c_str());
	}
}

int* GenSearchVect(int searchStep, int numSearches)
{
	int* sss = new int[numSearches];
	int curr = -searchStep * (numSearches >> 1);
	for (int it = 0; it < numSearches; ++it)
	{
		sss[it] = curr;
		curr += searchStep;
	}
	return sss;
}

unsigned char* GenRandImage(int rows, int cols)
{
	unsigned char* sss = new unsigned char[rows * cols];
	for (size_t it = 0; it < rows * cols; ++it)
	{
		float r = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
		sss[it] = (unsigned char)(r * 256);
	}
	return sss;
}

float* AllocFloats(size_t numFloats)
{
	float* ret = new float[numFloats];
	for (unsigned int it = 0; it < numFloats; ++it)
	{
		ret[it] = -2;
	}
	return ret;
}

typedef void (*calcrdmd_auxag)(unsigned char* im1, unsigned char* im2,
	int im_rows, int im_cols,
	const int* searchvec_x, int numSearchX, const int* searchvec_y, int numSearchY,
	int aggregationhalfwidy, int aggregationhalfheight,
	int aggregationdecimx, int aggregationdecimy,
	int similarity,
	float* out_dmd_ag, float* fitgrade_ag, long long* feat);

int main()
{
	PrintTimeStamp("begining");
	const int search_step_x = 4, search_step_y = 4;
	const int num_searches_x = 5, num_searches_y = 5;
	const int half_agg_y = 20, half_agg_x = 20;
	const int decim_x = 2, decim_y = 2;
	int* searchvec_x = GenSearchVect(search_step_x, num_searches_x);
	int* searchvec_y = GenSearchVect(search_step_y, num_searches_y);
	const int im_rows = 1000, im_cols = 1000;
	const int similarity = 1;
	PrintTimeStamp("Before generating random image");
	unsigned char* im1 = GenRandImage(im_rows, im_cols);
	unsigned char* im2 = GenRandImage(im_rows, im_cols);
	PrintTimeStamp("after generating random image");
	float* out_dmd_ag = AllocFloats(500 * 500 * 2);
	float* fitgrade_ag = AllocFloats(500 * 500);
	long long feat[20];
	PrintTimeStamp("Loading the DLL...");
	HINSTANCE hinstLib;
	calcrdmd_auxag ccc;
	BOOL fFreeResult, fRunTimeLinkSucceeded = FALSE;
	hinstLib = LoadLibrary("...\\...\\x64\\Release\\calcblockagdmd.dll");
	if (hinstLib == NULL)
	{
		std::cout << "-F- Fail to load library" << std::endl;
		return 1;
	}
	ccc = (calcrdmd_auxag)GetProcAddress(hinstLib, "calcrdmd_auxag");
	if (ccc == NULL)
	{
		std::cout << "-F- Manage to load the library but cannot find the function" << std::endl;
	}
	PrintTimeStamp("Before calling calcrdmd_auxag");
	(ccc)(im1, im2, im_rows, im_cols,
		searchvec_x, num_searches_x, searchvec_y, num_searches_y,
		half_agg_x, half_agg_y, decim_x, decim_y, similarity,
		out_dmd_ag, fitgrade_ag, feat);
	PrintTimeStamp("Free the library");
	fFreeResult = FreeLibrary(hinstLib);
	PrintTimeStamp("After calling calcrdmd_auxag");
	delete[] searchvec_x;
	delete[] searchvec_y;
	delete[] im1;
	delete[] im2;
	delete[] out_dmd_ag;
	delete[] fitgrade_ag;
	PrintTimeStamp("end");
	PrintAll();
	return 0;
}