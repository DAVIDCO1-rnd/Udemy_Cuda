
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm> //for std::min_element
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <string>
#include <chrono> 

#include <windows.h>

using namespace std::chrono;

using std::string;
using std::vector;

int HH_SIZE;
int KK_SIZE;

struct par_struct {
	int* hh;
	int* kk;
	int cc;
	int dd;
	int ee;
	int ff;
	int similarity;
};

int g2dim1 = 20;
int g2dim2 = 23;

int g1dim1 = 20;
int g1dim2 = 23;

//int g1dim1 = 2556;
//int g1dim2 = 9820;
//int g2dim1 = 2576;
//int g2dim2 = 9840;

string folder_path = "..\\..\\..\\MATLAB_MEX_CUDA\\bin_files";

float2* build_random_float2_matrix(int rows, int cols)
{
	float2* mat = new float2[rows*cols];
	int counter = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{			
			size_t currentIndex = j * rows + i; //this index is calculated for matlab (just like it is done for 3D matrix)
			float2 num;
			num.x = 0.7f + counter;
			num.y = 0.2f + counter;
			mat[currentIndex] = num;
			counter++;
		}
	}
	return mat;
}

float2* build_random_3D_matrix(int dimension1, int dimension2, int dimension3)
{
	float2* mat = new float2[dimension1*dimension2*dimension3];
	int counter = 0;
	for (int i = 0; i < dimension1; i++)
	{
		for (int j = 0; j < dimension2; j++)
		{
			for (int k = 0; k < dimension3; k++)
			{
				//size_t tempIndex = j * dimension3 + k;
				//size_t currentIndex = i * dimension2 * dimension3 + tempIndex;
				size_t currentIndex = k * dimension1 * dimension2 + j * dimension1 + i; //this index is calculated for matlab
				float2 num;
				num.x = 0.1 + counter;
				num.y = 0.8 + counter;				
				mat[currentIndex] = num;
				counter++;
			}
		}
	}
	return mat;
}

void load_variables(float2*& g1, float2*& g2, float2& aa, float2& bb, vector<int2>& mm, par_struct& par, int& iblk)
{
	iblk = 4;

	bb.x = 60;
	bb.y = 60;

	aa.x = 50;
	aa.y = 50;
	int2 mm0;
	mm0.x = 1621;
	mm0.y = 1229;

	int2 mm1;
	mm1.x = 3240;
	mm1.y = 1229;

	int2 mm2;
	mm2.x = 3240;
	mm2.y = 2456;

	int2 mm3;
	mm3.x = 1621;
	mm3.y = 2456;
	mm.push_back(mm0);
	mm.push_back(mm1);
	mm.push_back(mm2);
	mm.push_back(mm3);

	par.cc = 2;
	par.dd = 2;
	par.ee = 50;
	par.ff = 50;
	par.similarity = 5; // 1 / 5 / 6
	par.hh = new int[21]{ -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	par.kk = new int[21]{ -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	HH_SIZE = 21;
	KK_SIZE = 21;

	g1 = build_random_float2_matrix(g1dim1, g1dim2);
	g2 = build_random_float2_matrix(g2dim1, g2dim2);
}


__device__ float2 calc_var2_val_cuda(int i, int j, int k, int* dev_g2_rows, int* dev_col_min, int* dev_row_min, int* dev_ww, int* dev_min_x, int* dev_min_y, int* cols_step, int* rows_step, float2* dev_g2)
{
	int tx = (k / dev_ww[0]) + dev_min_x[0];
	int ty = (k % dev_ww[0]) + dev_min_y[0];

	int current_col = dev_col_min[0] + cols_step[0] * j;
	int current_var1 = current_col + tx - 1;

	int current_row = dev_row_min[0] + rows_step[0] * i;
	int current_var2 = current_row + ty;

	int currentIndex_matlab = current_var1 * dev_g2_rows[0] + current_var2;
	int currentIndex_for_g2 = currentIndex_matlab - 1;

	//float2 currentVal = dev_g2[currentIndex_for_g2];
	float2 currentVal;
	currentVal.x = (float)currentIndex_matlab;
	currentVal.y = (float)currentIndex_matlab;
	return currentVal;
}


float2 calc_var2_val_cpu(int i, int j, int k, int g2_rows, int col_min, int row_min, int ww, int min_x, int min_y, int cols_step, int rows_step)
{
	int tx = (k / ww) + min_x;
	int ty = (k % ww) + min_y;

	int current_col = col_min + cols_step * j;
	int current_var1 = current_col + tx - 1;

	int current_row = row_min + rows_step * i;
	int current_var2 = current_row + ty;

	int currentIndex_matlab = current_var1 * g2_rows + current_var2;
	int currentIndex_for_g2 = currentIndex_matlab - 1;

	//float2 currentVal = dev_g2[currentIndex_for_g2];
	float2 currentVal;
	currentVal.x = (float)currentIndex_matlab;
	currentVal.y = (float)currentIndex_matlab;
	return currentVal;
}

float2* build_var2_cpu(float2* g2, int col_min, int row_min, int pp, int ww, int min_x, int min_y, vector<int> dimensions, int jj, int rr, float2* g1, float2 aa, vector<int2> mm)
{
	int cols_step = jj;
	int rows_step = rr;
	size_t dimension3 = (size_t)(pp * ww);
	int dimension1 = dimensions[0];
	int dimension2 = dimensions[1];
	float2* var2 = new float2[dimension1*dimension2*dimension3];
	for (int i=0 ; i<dimension1 ; i++)
	{
		for (int j = 0; j < dimension2; j++)
		{
			for (int k = 0; k < dimension3; k++)
			{
				float2 var2_val = calc_var2_val_cpu(i, j, k, g2dim1, col_min, row_min, ww, min_x, min_y, cols_step, rows_step);
				size_t current_index_for_var2 = k * dimension1 * dimension2 + j * dimension1 + i;
				var2[current_index_for_var2] = var2_val;
			}
		}
	}
	return var2;
}

int calc_tg1_row_val(float2 aa, int2 mm_1, int2 mm_2, int ff, int cc, int i)
{
	float imag_aa = aa.y;
	int imag_mm_1 = mm_1.y;
	int par_ff = ff;
	int tg1_rows_first_part = imag_mm_1 - par_ff;
	int tg_rows_delta = cc;
	int imag_mm_2 = mm_2.y;
	int tg1_rows_third_part = imag_mm_2 + par_ff;

	int tg1_row_val = imag_aa + tg1_rows_first_part + i * tg_rows_delta;
	return tg1_row_val;
}

int calc_tg1_col_val(float2 aa, int2 mm_0, int2 mm_1, int ee, int dd, int j)
{
	float real_aa = aa.x;
	int real_mm_0 = mm_0.x;
	int tg1_cols_first_part = real_mm_0 - ee;
	int tg1_cols_delta = dd;
	int real_mm_1 = mm_1.x;
	int tg1_cols_third_part = real_mm_1 + ee;

	int tg1_col_val = real_aa + tg1_cols_first_part + j * tg1_cols_delta;
	return tg1_col_val;
}

__device__ int calc_tg1_row_val_cuda(float2 aa, int2 mm_1, int2 mm_2, int ff, int cc, int i)
{
	float imag_aa = aa.y;
	int imag_mm_1 = mm_1.y;
	int par_ff = ff;
	int tg1_rows_first_part = imag_mm_1 - par_ff;
	int tg_rows_delta = cc;
	//int imag_mm_2 = mm_2.y;
	//int tg1_rows_third_part = imag_mm_2 + par_ff;

	int tg1_row_val = imag_aa + tg1_rows_first_part + i * tg_rows_delta;
	return tg1_row_val;
}

__device__ int calc_tg1_col_val_cuda(float2 aa, int2 mm_0, int2 mm_1, int ee, int dd, int j)
{
	float real_aa = aa.x;
	int real_mm_0 = mm_0.x;
	int tg1_cols_first_part = real_mm_0 - ee;
	int tg1_cols_delta = dd;
	//int real_mm_1 = mm_1.x;
	//int tg1_cols_third_part = real_mm_1 + ee;

	int tg1_col_val = real_aa + tg1_cols_first_part + j * tg1_cols_delta;
	return tg1_col_val;
}

__device__ float2 calc_tg1_val_cuda(float2 aa, int2 mm_0, int2 mm_1, int2 mm_2, int cc, int dd, int ee, int ff, int i, int j, float2* g1, int* dev_g1dim1)
{
	int current_row_matlab = calc_tg1_row_val_cuda(aa, mm_1, mm_2, ff, cc, i);
	int current_col_matlab = calc_tg1_col_val_cuda(aa, mm_0, mm_1, ee, dd, j);
	int current_row = current_row_matlab - 1;
	int current_col = current_col_matlab - 1;
	int g1_current_index = current_col * dev_g1dim1[0] + current_row;
	float2 tg1_val = g1[g1_current_index];
	return tg1_val;
}

float2 calc_tg1_val(float2 aa, int2 mm_0, int2 mm_1, int2 mm_2, int cc, int dd, int ee, int ff, int i, int j, float2* g1)
{
	int current_row_matlab = calc_tg1_row_val(aa, mm_1, mm_2, ff, cc, i);
	int current_col_matlab = calc_tg1_col_val(aa, mm_0, mm_1, ee, dd, j);
	int current_row = current_row_matlab - 1;
	int current_col = current_col_matlab - 1;
	int g1_current_index = current_col * g1dim1 + current_row;
	float2 tg1_val = g1[g1_current_index];
	return tg1_val;
}

float calc_var10_val(int i, int j, int k, int col_min, int row_min, int ww, int min_x, int min_y, float2* g1, float2 aa, int2 mm_0, int2 mm_1, int2 mm_2, int cc, int dd, int ee, int ff, int similarity, int cols_step, int rows_step)
{
	float2 current_complex_val_tg1 = calc_tg1_val(aa, mm_0, mm_1, mm_2, cc, dd, ee, ff, i, j, g1);
	float2 current_complex_val_var2 = calc_var2_val_cpu(i, j, k, g2dim1, col_min, row_min, ww, min_x, min_y, cols_step, rows_step);
	float2 diff_val;
	diff_val.x = current_complex_val_tg1.x - current_complex_val_var2.x;
	diff_val.y = current_complex_val_tg1.y - current_complex_val_var2.y;
	float norm_diff = sqrt(diff_val.x * diff_val.x + diff_val.y * diff_val.y);
	float out_val;
	if (similarity == 1)
	{
		float norm_current_complex_val_tg1 = sqrt(current_complex_val_tg1.x * current_complex_val_tg1.x + current_complex_val_tg1.y * current_complex_val_tg1.y);
		float norm_current_complex_val_var2 = sqrt(current_complex_val_var2.x*current_complex_val_var2.x + current_complex_val_var2.y*current_complex_val_var2.y);
		out_val = 0.5f * (norm_current_complex_val_tg1 + norm_current_complex_val_var2) - norm_diff;
	}
	else
	{
		out_val = 1.0f - norm_diff;
	}
	out_val = diff_val.x * diff_val.x;
	return out_val;
}

float* build_var90(int col_min, int row_min, int ww, int min_x, int min_y, vector<int> dimensions, int cols_step, int rows_step, float2* g1, float2 aa, vector<int2> mm, par_struct par)
{
	int dim1 = dimensions[0];
	int dim2 = dimensions[1];
	int dim3 = dimensions[2];
	float* var10 = new float[dim1*dim2*dim3];
	int2 mm_0 = mm[0];
	int2 mm_1 = mm[1];
	int2 mm_2 = mm[2];
	


	for (int i = 0; i < dim1; i++)
	{
		for (int j = 0; j < dim2; j++)
		{
			for (int k = 0; k < dim3; k++)
			{
				int var10_current_index = k * dim1 * dim2 + j * dim1 + i;
				float current_val = calc_var10_val(i, j, k, col_min, row_min, ww, min_x, min_y, g1, aa, mm_0, mm_1, mm_2, par.cc, par.dd, par.ee, par.ff, par.similarity, cols_step, rows_step);
				var10[var10_current_index] = current_val;
			}
		}
	}
	return var10;
}

float* build_var90_cuda(float2* g1, float2 aa, int col_min, int row_min, int ww, int min_x, int min_y, int dim1, int dim2, int dim3, int cols_step, int rows_step, int cc, int dd, int ee, int ff, int similarity, int2 mm_0, int2 mm_1, int2 mm_2)
{
	float* var10 = new float[dim1*dim2*dim3];
	/*for (int i = 0; i < dim1; i++)*/
	int i = 0;
	while (i<dim1)
	{
		for (int j = 0; j < dim2; j++)
		{
			for (int k = 0; k < dim3; k++)
			{
				int var10_current_index = k * dim1 * dim2 + j * dim1 + i;
				float current_val = calc_var10_val(i, j, k, col_min, row_min, ww, min_x, min_y, g1, aa, mm_0, mm_1, mm_2, cc, dd, ee, ff, similarity, cols_step, rows_step);
				var10[var10_current_index] = current_val;
			}
		}
		i++;
	}
	return var10;
}



__device__ float calc_var10_val_cuda(float* dev_var10, float2* dev_g1, float2* dev_g2, int i, int j, int k, float2* dev_aa, int* dev_col_min, int* dev_row_min, int* dev_ww, int* dev_min_x, int* dev_min_y, int* dev_dim1, int* dev_dim2, int* dev_dim3, int* dev_cols_step, int* dev_rows_step, int* dev_ee, int* dev_ff, int* dev_similarity, int* dev_g1dim1, int* dev_g2dim1, int2* dev_mm0, int2* dev_mm1, int2* dev_mm2)
{
	float2 current_complex_val_tg1 = calc_tg1_val_cuda(dev_aa[0], dev_mm0[0], dev_mm1[0], dev_mm2[0], dev_rows_step[0], dev_cols_step[0], dev_ee[0], dev_ff[0], i, j, dev_g1, dev_g1dim1);
	float2 current_complex_val_var2 = calc_var2_val_cuda(i, j, k, dev_g2dim1, dev_col_min, dev_row_min, dev_ww, dev_min_x, dev_min_y, dev_cols_step, dev_rows_step, dev_g2);
	float2 diff_val;
	diff_val.x = current_complex_val_tg1.x - current_complex_val_var2.x;
	diff_val.y = current_complex_val_tg1.y - current_complex_val_var2.y;
	float norm_diff = sqrt(diff_val.x * diff_val.x + diff_val.y * diff_val.y);
	float out_val;
	if (dev_similarity[0] == 1)
	{
		float norm_current_complex_val_tg1 = sqrt(current_complex_val_tg1.x * current_complex_val_tg1.x + current_complex_val_tg1.y * current_complex_val_tg1.y);
		float norm_current_complex_val_var2 = sqrt(current_complex_val_var2.x*current_complex_val_var2.x + current_complex_val_var2.y*current_complex_val_var2.y);
		out_val = 0.5f * (norm_current_complex_val_tg1 + norm_current_complex_val_var2) - norm_diff;
	}
	else
	{
		out_val = 1.0f - norm_diff;
	}
	out_val = diff_val.x * diff_val.x;
	return out_val;
}


__global__ void cumsum_dim_1_cuda(float* dev_var10, int* dev_dim1, int* dev_dim2, int* dev_dim3)
{
	int j = threadIdx.x;
	while (j < dev_dim2[0])
	{
		int k = blockIdx.x;
		while (k < dev_dim3[0])
		{
			for (int i = 0; i < dev_dim1[0]; i++)
			{
				int var10_current_index = k * dev_dim1[0] * dev_dim2[0] + j * dev_dim1[0] + i;

				if (i == 0)
				{
					dev_var10[var10_current_index] = dev_var10[var10_current_index];
				}
				else
				{
					int var20_prev_index = k * dev_dim1[0] * dev_dim2[0] + j * dev_dim1[0] + i - 1;
					dev_var10[var10_current_index] = dev_var10[var20_prev_index] + dev_var10[var10_current_index];
				}
			}
			k += gridDim.x;
		}
		j += blockDim.x;
	}
}

__global__ void cumsum_dim_2_cuda(float* dev_var10, int* dev_dim1, int* dev_dim2, int* dev_dim3)
{
	int i = threadIdx.x;
	while (i < dev_dim1[0])
	{
		int k = blockIdx.x;
		while (k < dev_dim3[0])
		{
			for (int j = 0; j < dev_dim2[0]; j++)
			{
				int var10_current_index = k * dev_dim1[0] * dev_dim2[0] + j * dev_dim1[0] + i;

				if (j == 0)
				{
					dev_var10[var10_current_index] = dev_var10[var10_current_index];
				}
				else
				{
					int prev_index_j = k * dev_dim1[0] * dev_dim2[0] + (j - 1) * dev_dim1[0] + i;
					dev_var10[var10_current_index] = dev_var10[prev_index_j] + dev_var10[var10_current_index];
				}
			}
			k += gridDim.x;
		}
		i += blockDim.x;
	}
}

__global__ void build_var40_cuda(float* dev_var40, float* dev_var10, int* dev_dim1, int* dev_dim2, int* dev_dim3, int* dev_cc, int* dev_dd, int* dev_ee, int* dev_ff, int2* dev_mm0, int2* dev_mm1, int2* dev_mm2)
{
	int nr = (dev_mm2[0].y - dev_mm1[0].y + 1) / dev_cc[0];
	int nc = (dev_mm1[0].x - dev_mm0[0].x + 1) / dev_dd[0];

	int ax = 2 * dev_ee[0] / dev_dd[0];
	int ay = 2 * dev_ff[0] / dev_cc[0];
	int axy = ax * ay;

	int i = threadIdx.x;
	while (i < nr)
	{
		int j = blockIdx.x;
		while (j < nc)
		{
			for (int k = 0; k < dev_dim3[0]; k++)
			{
				int index1 = k * nr * nc + j * nr + i;
				int index2 = k * dev_dim1[0] * dev_dim2[0] + j * dev_dim1[0] + i;
				int index3 = k * dev_dim1[0] * dev_dim2[0] + (j + ax) * dev_dim1[0] + (i + ay);
				int index4 = k * dev_dim1[0] * dev_dim2[0] + (j + ax) * dev_dim1[0] + i;
				int index5 = k * dev_dim1[0] * dev_dim2[0] + j * dev_dim1[0] + (i + ay);
				float currentVal = (dev_var10[index2] + dev_var10[index3] - dev_var10[index4] - dev_var10[index5]) / axy;
				dev_var40[index1] = currentVal;
			}
			j += gridDim.x;
		}
		i += blockDim.x;
	}
}

__global__ void build_max_third_dimension_cuda(float* dev_var40, float* dev_var40_max_values, unsigned short int* dev_var40_indexes_of_max_values, int dev_dim1, int dev_dim2, int* dev_dim3)
{
	int i = threadIdx.x;
	while (i < dev_dim1)
	{
		int j = blockIdx.x;
		while (j < dev_dim2)
		{
			int current_index_for_2D_matrix = j * dev_dim1 + i;
			for (unsigned short int k = 0; k < dev_dim3[0]; k++)
			{
				int current_index_for_3D_matrix = k * dev_dim1 * dev_dim2 + j * dev_dim1 + i;				
				float current_val = dev_var40[current_index_for_3D_matrix];
				if (k == 0 || current_val > dev_var40_max_values[current_index_for_2D_matrix])
				{
					dev_var40_max_values[current_index_for_2D_matrix] = current_val;
					dev_var40_indexes_of_max_values[current_index_for_2D_matrix] = k;
				}
			}
			j += gridDim.x;
		}
		i += blockDim.x;
	}
}

__global__ void build_var50_cuda(float2* dev_var50, int* dev_kk, int* dev_kk_size, int* dev_hh, int* dev_hh_size, unsigned short int* matrix_indexes_of_max_values, int nr, int nc, int* dev_dim3)
{
	int i = threadIdx.x;
	while (i < nr)
	{
		int j = blockIdx.x;
		while (j < nc)
		{
			int current_index = j * nr + i;
			//int current_index_for_xysearch_matlab = matrix_indexes_of_max_values[current_index];
			//int current_index_for_xysearch = current_index_for_xysearch_matlab - 1;
			unsigned short int current_index_for_xysearch = matrix_indexes_of_max_values[current_index];
			unsigned short int xsearch_index = current_index_for_xysearch / dev_kk_size[0];
			unsigned short int ysearch_index = current_index_for_xysearch % dev_kk_size[0];
			float xsearch_val = dev_hh[xsearch_index];
			float ysearch_val = dev_kk[ysearch_index];
			float2 current_val;
			current_val.x = xsearch_val;
			current_val.y = ysearch_val;
			dev_var50[current_index] = current_val;
			j += gridDim.x;
		}
		i += blockDim.x;
	}
}

__global__ void build_var60_cuda(float* dev_var60, float* dev_var40, unsigned short int* dev_var40_indexes_of_max_values, int nr, int nc, int* dev_dim3, int* dev_kk_size)
{
	int nsearchy = dev_kk_size[0];
	bool is1dsearch = nsearchy == 1;
	int length_small_arr = 3; //length of the array {-1, 0, 1} or { -nsearchy, 0, nsearchy }
	float very_small_number = -999999;

	int di_length;
	if (is1dsearch)
	{
		di_length = length_small_arr;
	}
	else
	{
		di_length = length_small_arr * length_small_arr;
	}

	int var40_mult_dimensions = nr * nc * dev_dim3[0];
	int i = threadIdx.x;
	while (i < nr)
	{
		int j = blockIdx.x;
		while (j < nc)
		{
			for (int k = 0; k < di_length; k++)
			{
				int di_val;
				if (is1dsearch)
				{
					di_val = k - 1; //values - 1, 0, 1
				}
				else
				{
					int arr1_index = k / length_small_arr; // indexes 0, 1, 2
					int arr1_val = arr1_index - 1; //values - 1, 0, 1
					int arr2_index = k % length_small_arr; //indexes 0, 1, 2
					int arr2_val = (arr2_index - 1) * nsearchy; //values - 5, 0, 5
					di_val = arr1_val + arr2_val;
				}
	
				int current_index_var40_indexes = j * nr + i;
				unsigned short int var40_current_index = dev_var40_indexes_of_max_values[current_index_var40_indexes];
				unsigned short int var40_current_index_matlab = var40_current_index + 1;
				int current_val = (int)var40_current_index_matlab + di_val;
				int val1 = (current_val - 1)*nr * nc;
				int current_index_for_xx_yy = j * nr + i;
	
				int xx_val = (current_index_for_xx_yy / nr) + 1;
				int yy_val = (current_index_for_xx_yy % nr) + 1;
	
				int val2 = (xx_val - 1)*nr;
				int intermediate_lind3_val = val1 + val2 + yy_val;
				int lind3_val_matlab;
				if (intermediate_lind3_val <= 0 || intermediate_lind3_val > var40_mult_dimensions)
				{
					lind3_val_matlab = 1;
				}
				else
				{
					lind3_val_matlab = intermediate_lind3_val;
				}
	
				int lind3_val = lind3_val_matlab - 1;
				int current_index_for_var60 = k * nr * nc + j * nr + i;
				
				float current_var60_val;
				if (lind3_val == 0)
				{
					current_var60_val = very_small_number;
				}
				else
				{
					current_var60_val = dev_var40[lind3_val];
				}
				dev_var60[current_index_for_var60] = current_var60_val;
			}
			j += gridDim.x;
		}
		i += blockDim.x;
	}
}

__device__ float calc_sft_val_from_vector_cuda(float* mean_vector)
{
	float a = mean_vector[0];
	float b = mean_vector[1];
	float c = mean_vector[2];
	float denominator = a + c - 2 * b;
	float abs_denominator = abs(denominator);
	float eps = 0.0000001;
	float sft_val;
	if (abs_denominator < eps)
	{
		sft_val = 0;
	}
	else
	{
		sft_val = ((a - c) / 2) / denominator;
	}


	if (sft_val < -0.5)
	{
		sft_val = -0.5;
	}

	if (sft_val > 0.5)
	{
		sft_val = 0.5;
	}

	return sft_val;
}

__device__ float calc_sft_val_from_matrix_cuda(float* matrix, int i, int j, int dim1, int dim2, int dim)
{
	int dim3 = 3;
	int dim4 = 3;
	float sft_val = 0;
	if (dim == 4)
	{
		float* mean_vector = new float[dim3];
		for (int k = 0; k < dim3; k++)
		{
			float current_sum = 0;
			for (int w = 0; w < dim4; w++)
			{
				int current_index = w * dim3*dim2*dim1 + k * dim2*dim1 + j * dim1 + i;
				current_sum = current_sum + matrix[current_index];
			}
			float current_mean = current_sum / dim4;
			mean_vector[k] = current_mean;
		}
		sft_val = calc_sft_val_from_vector_cuda(mean_vector);
		delete[] mean_vector;
	}

	if (dim == 3)
	{
		float* mean_vector = new float[dim4];
		for (int w = 0; w < dim4; w++)
		{
			float current_sum = 0;
			for (int k = 0; k < dim3; k++)
			{
				int current_index = w * dim3*dim2*dim1 + k * dim2*dim1 + j * dim1 + i;
				current_sum = current_sum + matrix[current_index];
			}
			float current_mean = current_sum / dim3;
			mean_vector[w] = current_mean;
		}
		sft_val = calc_sft_val_from_vector_cuda(mean_vector);
		delete[] mean_vector;
	}

	return sft_val;
}

__global__ void build_var70_cuda(float2* dev_var70, float2* dev_var50, float* dev_var60, int nr, int nc, int* dev_kk, int* dev_kk_size, int* dev_hh)
{
	int dsearchx = abs(dev_hh[1] - dev_hh[0]);
	int dsearchy = abs(dev_kk[1] - dev_kk[0]);
	int nsearchy = dev_kk_size[0];
	bool is1dsearch = nsearchy == 1;

	int i = threadIdx.x;
	while (i < nr)
	{
		int j = blockIdx.x;
		while (j < nc)
		{
			int current_index = j * nr + i;
			float2 dev_var50_current_val = dev_var50[current_index];
			if (is1dsearch == false)
			{
				float sftx_current_val = calc_sft_val_from_matrix_cuda(dev_var60, i, j, nr, nc, 4);
				float sfty_current_val = calc_sft_val_from_matrix_cuda(dev_var60, i, j, nr, nc, 3);
				float2 var70_current_val;
				var70_current_val.x = sftx_current_val * dsearchx + dev_var50_current_val.x;
				var70_current_val.y = sfty_current_val * dsearchy + dev_var50_current_val.y;
				dev_var70[current_index] = var70_current_val;
			}
			else
			{
				int index1 = 0 * nr * nc + j * nr + i;
				int index2 = 1 * nr * nc + j * nr + i;
				int index3 = 2 * nr * nc + j * nr + i;
				float a = dev_var60[index1];
				float b = dev_var60[index2];
				float c = dev_var60[index3];
				float* vec = new float[3]{ a,b,c };
				float sftx_current_val = calc_sft_val_from_vector_cuda(vec);
				delete[] vec;
				float2 var70_current_val;
				var70_current_val.x = sftx_current_val * dsearchx + dev_var50_current_val.x;
				var70_current_val.y = 0 + dev_var50_current_val.y;
				dev_var70[current_index] = var70_current_val;
			}
			j += gridDim.x;
		}
		i += blockDim.x;
	}
}

__global__ void build_var80_cuda(float2* dev_var80, float2* dev_var50, float2* dev_var70, int nr, int nc)
{
	int i = threadIdx.x;
	while (i < nr)
	{
		int j = blockIdx.x;
		while (j < nc)
		{
			int current_index = j * nr + i;
			dev_var80[current_index].x = dev_var50[current_index].x + dev_var70[current_index].x;
			dev_var80[current_index].y = dev_var50[current_index].y + dev_var70[current_index].y;
			j += gridDim.x;
		}
		i += blockDim.x;
	}
}



__global__ void calc_var90_matrix_cuda(float* dev_var10, float2* dev_g1, float2* dev_g2, float2* dev_aa, int* dev_col_min, int* dev_row_min, int* dev_ww, int* dev_min_x, int* dev_min_y, int* dev_dim1, int* dev_dim2, int* dev_dim3, int* dev_cols_step, int* dev_rows_step, int* dev_ee, int* dev_ff, int* dev_similarity, int* dev_g1dim1, int* dev_g2dim1, int2* dev_mm0, int2* dev_mm1, int2* dev_mm2)
{
	int i = threadIdx.x;
	while (i < dev_dim1[0])
	{
		int j = blockIdx.x;
		while (j < dev_dim2[0])
		{
			for (int k = 0; k < dev_dim3[0]; k++)
			{
				int var10_current_index = k * dev_dim1[0] * dev_dim2[0] + j * dev_dim1[0] + i;
				float var10_val = calc_var10_val_cuda(dev_var10, dev_g1, dev_g2, i, j, k, dev_aa, dev_col_min, dev_row_min, dev_ww, dev_min_x, dev_min_y, dev_dim1, dev_dim2, dev_dim3, dev_cols_step, dev_rows_step, dev_ee, dev_ff, dev_similarity, dev_g1dim1, dev_g2dim1, dev_mm0, dev_mm1, dev_mm2);
				dev_var10[var10_current_index] = var10_val;
			}
			j += gridDim.x;
		}
		i += blockDim.x;
	}
}

void write_data(string folderPath, string filename, int num_of_bytes, void* var)
{
	string fileFullPath = folderPath + "/" + filename;
	char* var2_bytes = (char*)var;
	std::ofstream varFile(fileFullPath, std::ios::out | std::ios::binary);
	if (varFile.is_open())
	{
		varFile.write(var2_bytes, num_of_bytes);
	}
	varFile.close();
}

void build_var90_before_cuda(int col_min, int row_min, int ww, int min_x, int min_y, vector<int> dimensions, float2* g1, float2* g2, float2 aa, vector<int2> mm, int cc, int dd, int ee, int ff, int similarity, int* kk, int* hh, float*& var40_max_values, float2*& var70)
{
	int dimension1 = dimensions[0];
	int dimension2 = dimensions[1];
	int dimension3 = dimensions[2];
	int2 mm0 = mm[0];
	int2 mm1 = mm[1];
	int2 mm2 = mm[2];

	float* dev_var10 = NULL;
	unsigned int var10_num_of_elements = dimension1 * dimension2 * dimension3;
	size_t var10_num_of_bytes = var10_num_of_elements * sizeof(float);
	cudaError_t cudaStatus_var10_alloc = cudaMalloc((void**)&dev_var10, var10_num_of_bytes);
	if (cudaStatus_var10_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_g1
	float2* dev_g1 = NULL;
	unsigned int g1_num_of_elements = g1dim1 * g1dim2;
	size_t g1_num_of_bytes = g1_num_of_elements * sizeof(float2);
	cudaError_t cudaStatus_g1_alloc = cudaMalloc((void**)&dev_g1, g1_num_of_bytes);
	if (cudaStatus_g1_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_g1_memcpy = cudaMemcpy(dev_g1, g1, g1_num_of_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_g1_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_g2
	float2* dev_g2 = NULL;
	unsigned int g2_num_of_elements = g2dim1 * g2dim2;
	size_t g2_num_of_bytes = g2_num_of_elements * sizeof(float2);
	cudaError_t cudaStatus_g2_alloc = cudaMalloc((void**)&dev_g2, g2_num_of_bytes);
	if (cudaStatus_g2_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_g2_memcpy = cudaMemcpy(dev_g2, g2, g2_num_of_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_g2_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		//create dev_kk
	int* dev_kk = NULL;
	size_t kk_bytes = sizeof(int) * KK_SIZE;
	cudaError_t cudaStatus_kk_alloc = cudaMalloc((void**)&dev_kk, kk_bytes);
	if (cudaStatus_kk_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_kk_memcpy = cudaMemcpy(dev_kk, kk, kk_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_kk_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//create dev_kk_size
	int* dev_kk_size = NULL;
	size_t kk_size_bytes = sizeof(int);
	cudaError_t cudaStatus_kk_size_alloc = cudaMalloc((void**)&dev_kk_size, kk_size_bytes);
	if (cudaStatus_kk_size_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_kk_size_memcpy = cudaMemcpy(dev_kk_size, &KK_SIZE, kk_size_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_kk_size_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//create dev_hh
	int* dev_hh = NULL;
	size_t hh_bytes = sizeof(int) * HH_SIZE;
	cudaError_t cudaStatus_hh_alloc = cudaMalloc((void**)&dev_hh, hh_bytes);
	if (cudaStatus_hh_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_hh_memcpy = cudaMemcpy(dev_hh, hh, hh_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_hh_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//create dev_hh_size
	int* dev_hh_size = NULL;
	size_t hh_size_bytes = sizeof(int);
	cudaError_t cudaStatus_hh_size_alloc = cudaMalloc((void**)&dev_hh_size, hh_size_bytes);
	if (cudaStatus_hh_size_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_hh_size_memcpy = cudaMemcpy(dev_hh_size, &HH_SIZE, hh_size_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_hh_size_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




		//create dev_aa
	float2* dev_aa = NULL;
	size_t aa_bytes = sizeof(float2);
	cudaError_t cudaStatus_aa_alloc = cudaMalloc((void**)&dev_aa, aa_bytes);
	if (cudaStatus_aa_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_aa_memcpy = cudaMemcpy(dev_aa, &aa, aa_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_aa_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//create dev_col_min
	int* dev_col_min = NULL;
	size_t col_min_bytes = sizeof(int);
	cudaError_t cudaStatus_col_min_alloc = cudaMalloc((void**)&dev_col_min, col_min_bytes);
	if (cudaStatus_col_min_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_col_min_memcpy = cudaMemcpy(dev_col_min, &col_min, col_min_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_col_min_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//create dev_row_min
	int* dev_row_min = NULL;
	size_t row_min_bytes = sizeof(int);
	cudaError_t cudaStatus_row_min_alloc = cudaMalloc((void**)&dev_row_min, row_min_bytes);
	if (cudaStatus_row_min_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_row_min_memcpy = cudaMemcpy(dev_row_min, &row_min, row_min_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_row_min_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_ww
	int* dev_ww = NULL;
	size_t ww_bytes = sizeof(int);
	cudaError_t cudaStatus_ww_alloc = cudaMalloc((void**)&dev_ww, ww_bytes);
	if (cudaStatus_ww_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_ww_memcpy = cudaMemcpy(dev_ww, &ww, ww_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_ww_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_min_x
	int* dev_min_x = NULL;
	size_t min_x_bytes = sizeof(int);
	cudaError_t cudaStatus_min_x_alloc = cudaMalloc((void**)&dev_min_x, min_x_bytes);
	if (cudaStatus_min_x_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_min_x_memcpy = cudaMemcpy(dev_min_x, &min_x, min_x_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_min_x_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_min_
	int* dev_min_y = NULL;
	size_t min_y_bytes = sizeof(int);
	cudaError_t cudaStatus_min_y_alloc = cudaMalloc((void**)&dev_min_y, min_y_bytes);
	if (cudaStatus_min_y_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_min_y_memcpy = cudaMemcpy(dev_min_y, &min_y, min_y_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_min_y_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_dim1
	int* dev_dim1 = NULL;
	size_t dim1_bytes = sizeof(int);
	cudaError_t cudaStatus_dim1_alloc = cudaMalloc((void**)&dev_dim1, dim1_bytes);
	if (cudaStatus_dim1_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_dim1_memcpy = cudaMemcpy(dev_dim1, &dimension1, dim1_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_dim1_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_dim2
	int* dev_dim2 = NULL;
	size_t dim2_bytes = sizeof(int);
	cudaError_t cudaStatus_dim2_alloc = cudaMalloc((void**)&dev_dim2, dim2_bytes);
	if (cudaStatus_dim2_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_dim2_memcpy = cudaMemcpy(dev_dim2, &dimension2, dim2_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_dim2_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_dim3
	int* dev_dim3 = NULL;
	size_t dim3_bytes = sizeof(int);
	cudaError_t cudaStatus_dim3_alloc = cudaMalloc((void**)&dev_dim3, dim3_bytes);
	if (cudaStatus_dim3_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_dim3_memcpy = cudaMemcpy(dev_dim3, &dimension3, dim3_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_dim3_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//	//create dev_cols_step
	//int* dev_cols_step = NULL;
	//size_t cols_step_bytes = sizeof(int);
	//cudaError_t cudaStatus_cols_step_alloc = cudaMalloc((void**)&dev_cols_step, cols_step_bytes);
	//if (cudaStatus_cols_step_alloc != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	//free_arrays;
	//	return;
	//}
	//// Copy input vectors from host memory to GPU buffers.
	//cudaError_t cudaStatus_cols_step_memcpy = cudaMemcpy(dev_cols_step, &cols_step, cols_step_bytes, cudaMemcpyHostToDevice);
	//if (cudaStatus_cols_step_memcpy != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	//free_arrays;
	//	return;
	//}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//	//create dev_rows_step
	//int* dev_rows_step = NULL;
	//size_t rows_step_bytes = sizeof(int);
	//cudaError_t cudaStatus_rows_step_alloc = cudaMalloc((void**)&dev_rows_step, rows_step_bytes);
	//if (cudaStatus_rows_step_alloc != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	//free_arrays;
	//	return;
	//}
	//// Copy input vectors from host memory to GPU buffers.
	//cudaError_t cudaStatus_rows_step_memcpy = cudaMemcpy(dev_rows_step, &rows_step, rows_step_bytes, cudaMemcpyHostToDevice);
	//if (cudaStatus_rows_step_memcpy != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	//free_arrays;
	//	return;
	//}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_cc
	int* dev_cc = NULL;
	size_t cc_bytes = sizeof(int);
	cudaError_t cudaStatus_cc_alloc = cudaMalloc((void**)&dev_cc, cc_bytes);
	if (cudaStatus_cc_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_cc_memcpy = cudaMemcpy(dev_cc, &cc, cc_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_cc_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_dd
	int* dev_dd = NULL;
	size_t dd_bytes = sizeof(int);
	cudaError_t cudaStatus_dd_alloc = cudaMalloc((void**)&dev_dd, dd_bytes);
	if (cudaStatus_dd_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_dd_memcpy = cudaMemcpy(dev_dd, &dd, dd_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_dd_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_ee
	int* dev_ee = NULL;
	size_t ee_bytes = sizeof(int);
	cudaError_t cudaStatus_ee_alloc = cudaMalloc((void**)&dev_ee, ee_bytes);
	if (cudaStatus_ee_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_ee_memcpy = cudaMemcpy(dev_ee, &ee, ee_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_ee_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_ff
	int* dev_ff = NULL;
	size_t ff_bytes = sizeof(int);
	cudaError_t cudaStatus_ff_alloc = cudaMalloc((void**)&dev_ff, ff_bytes);
	if (cudaStatus_ff_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_ff_memcpy = cudaMemcpy(dev_ff, &ff, ff_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_ff_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_similarity
	int* dev_similarity = NULL;
	size_t similarity_bytes = sizeof(int);
	cudaError_t cudaStatus_similarity_alloc = cudaMalloc((void**)&dev_similarity, similarity_bytes);
	if (cudaStatus_similarity_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_similarity_memcpy = cudaMemcpy(dev_similarity, &similarity, similarity_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_similarity_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_mm0
	int2* dev_mm0 = NULL;
	size_t mm0_bytes = sizeof(int2);
	cudaError_t cudaStatus_mm0_alloc = cudaMalloc((void**)&dev_mm0, mm0_bytes);
	if (cudaStatus_mm0_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_mm0_memcpy = cudaMemcpy(dev_mm0, &mm0, mm0_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_mm0_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_mm1
	int2* dev_mm1 = NULL;
	size_t mm1_bytes = sizeof(int2);
	cudaError_t cudaStatus_mm1_alloc = cudaMalloc((void**)&dev_mm1, mm1_bytes);
	if (cudaStatus_mm1_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_mm1_memcpy = cudaMemcpy(dev_mm1, &mm1, mm1_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_mm1_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//create dev_mm2
	int2* dev_mm2 = NULL;
	size_t mm2_bytes = sizeof(int2);
	cudaError_t cudaStatus_mm2_alloc = cudaMalloc((void**)&dev_mm2, mm2_bytes);
	if (cudaStatus_mm2_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_mm2_memcpy = cudaMemcpy(dev_mm2, &mm2, mm2_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_mm2_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//create dev_g1dim1
	int* dev_g1dim1 = NULL;
	size_t g1dim1_bytes = sizeof(int);
	cudaError_t cudaStatus_g1dim1_alloc = cudaMalloc((void**)&dev_g1dim1, g1dim1_bytes);
	if (cudaStatus_g1dim1_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_g1dim1_memcpy = cudaMemcpy(dev_g1dim1, &g1dim1, g1dim1_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_g1dim1_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//create dev_g2dim1
	int* dev_g2dim1 = NULL;
	size_t g2dim1_bytes = sizeof(int);
	cudaError_t cudaStatus_g2dim1_alloc = cudaMalloc((void**)&dev_g2dim1, g2dim1_bytes);
	if (cudaStatus_g2dim1_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus_g2dim1_memcpy = cudaMemcpy(dev_g2dim1, &g2dim1, g2dim1_bytes, cudaMemcpyHostToDevice);
	if (cudaStatus_g2dim1_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//free_arrays;
		return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	


	int threadsPerBlock_var10 = 256;
	int blocksPerGrid_var10 = 256;

	calc_var90_matrix_cuda << < blocksPerGrid_var10, threadsPerBlock_var10 >> > (dev_var10, dev_g1, dev_g2, dev_aa, dev_col_min, dev_row_min, dev_ww, dev_min_x, dev_min_y, dev_dim1, dev_dim2, dev_dim3, dev_dd, dev_cc, dev_ee, dev_ff, dev_similarity, dev_g1dim1, dev_g2dim1, dev_mm0, dev_mm1, dev_mm2);


	//float* dev_var20 = NULL;
	//unsigned int var20_num_of_elements = dimension1 * dimension2 * dimension3;
	//size_t var20_num_of_bytes = var20_num_of_elements * sizeof(float);
	//cudaError_t cudaStatus_var20_alloc = cudaMalloc((void**)&dev_var20, var20_num_of_bytes);
	//if (cudaStatus_var20_alloc != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	//free_arrays;
	//	return;
	//}

	int threadsPerBlock_var20 = 256;
	int blocksPerGrid_var20 = 256;

	cumsum_dim_1_cuda << < blocksPerGrid_var20, threadsPerBlock_var20 >> > (dev_var10, dev_dim1, dev_dim2, dev_dim3);

	int threadsPerBlock_var30 = 256;
	int blocksPerGrid_var30 = 256;

	cumsum_dim_2_cuda << < blocksPerGrid_var30, threadsPerBlock_var30 >> > (dev_var10, dev_dim1, dev_dim2, dev_dim3);


	int nr = (mm2.y - mm1.y + 1) / cc;
	int nc = (mm1.x - mm0.x + 1) / dd;

	int threadsPerBlock_var40 = 256;
	int blocksPerGrid_var40 = 256;

	float* dev_var40 = NULL;
	unsigned int var40_num_of_elements = nr * nc * dimension3;
	size_t var40_num_of_bytes = var40_num_of_elements * sizeof(float);
	cudaError_t cudaStatus_var40_alloc = cudaMalloc((void**)&dev_var40, var40_num_of_bytes);
	if (cudaStatus_var40_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}

	build_var40_cuda << < blocksPerGrid_var40, threadsPerBlock_var40 >> > (dev_var40, dev_var10, dev_dim1, dev_dim2, dev_dim3, dev_cc, dev_dd, dev_ee, dev_ff, dev_mm0, dev_mm1, dev_mm2);

	int threadsPerBlock_var40_max_values_and_integers = 256;
	int blocksPerGrid_var40_max_values_and_integers = 256;

	float* dev_var40_max_values = NULL;
	unsigned int var40_max_values_num_of_elements = nr * nc;
	size_t var40_max_values_num_of_bytes = var40_max_values_num_of_elements * sizeof(float);
	cudaError_t cudaStatus_var40_max_values_alloc = cudaMalloc((void**)&dev_var40_max_values, var40_max_values_num_of_bytes);
	if (cudaStatus_var40_max_values_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}

	unsigned short int* dev_var40_indexes_of_max_values = NULL;
	unsigned int var40_indexes_of_max_values_num_of_elements = nr * nc;
	size_t var40_indexes_of_max_values_num_of_bytes = var40_indexes_of_max_values_num_of_elements * sizeof(unsigned short int);
	cudaError_t cudaStatus_var40_indexes_of_max_values_alloc = cudaMalloc((void**)&dev_var40_indexes_of_max_values, var40_indexes_of_max_values_num_of_bytes);
	if (cudaStatus_var40_indexes_of_max_values_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}

	build_max_third_dimension_cuda << < blocksPerGrid_var40_max_values_and_integers, threadsPerBlock_var40_max_values_and_integers >> > (dev_var40, dev_var40_max_values, dev_var40_indexes_of_max_values, nr, nc, dev_dim3);


	float2* dev_var50 = NULL;
	unsigned int var50_num_of_elements = nr * nc;
	size_t var50_num_of_bytes = var50_num_of_elements * sizeof(float2);
	cudaError_t cudaStatus_var50_alloc = cudaMalloc((void**)&dev_var50, var50_num_of_bytes);
	if (cudaStatus_var50_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}


	int threadsPerBlock_var50 = 256;
	int blocksPerGrid_var50 = 256;

	build_var50_cuda << < blocksPerGrid_var50, threadsPerBlock_var50 >> > (dev_var50, dev_kk, dev_kk_size, dev_hh, dev_hh_size, dev_var40_indexes_of_max_values, nr, nc, dev_dim3);

	int di_length = (KK_SIZE == 1) ? 3 : 9;
	float* dev_var60 = NULL;
	unsigned int var60_num_of_elements = nr * nc * di_length;
	size_t var60_num_of_bytes = var60_num_of_elements * sizeof(float);
	cudaError_t cudaStatus_var60_alloc = cudaMalloc((void**)&dev_var60, var60_num_of_bytes);
	if (cudaStatus_var60_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}


	int threadsPerBlock_var60 = 256;
	int blocksPerGrid_var60 = 256;
	build_var60_cuda << < blocksPerGrid_var60, threadsPerBlock_var60 >> > (dev_var60, dev_var40, dev_var40_indexes_of_max_values, nr, nc, dev_dim3, dev_kk_size);



	float2* dev_var70 = NULL;
	unsigned int var70_num_of_elements = nr * nc;
	size_t var70_num_of_bytes = var70_num_of_elements * sizeof(float2);
	cudaError_t cudaStatus_var70_alloc = cudaMalloc((void**)&dev_var70, var70_num_of_bytes);
	if (cudaStatus_var70_alloc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//free_arrays;
		return;
	}

	int threadsPerBlock_var70 = 256;
	int blocksPerGrid_var70 = 256;

	build_var70_cuda << < blocksPerGrid_var70, threadsPerBlock_var70 >> > (dev_var70, dev_var50, dev_var60, nr, nc, dev_kk, dev_kk_size, dev_hh);







	// Check for any errors launching the kernel
	cudaError_t cudaStatusLastError = cudaGetLastError();
	if (cudaStatusLastError != cudaSuccess) {
		fprintf(stderr, "getValsByIndexes launch failed: %s\n", cudaGetErrorString(cudaStatusLastError));
		//free_arrays
		return;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaError_t cudaStatusDeviceSynchronize = cudaDeviceSynchronize();
	if (cudaStatusDeviceSynchronize != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getValsByIndexes!\n", cudaStatusDeviceSynchronize);
		//free_arrays
		return;
	}



	// Copy output vector from GPU buffer to host memory.
	float* var10 = (float*)malloc(var10_num_of_bytes);
	cudaError_t cudaStatus_var10_memcpy = cudaMemcpy(var10, dev_var10, var10_num_of_bytes, cudaMemcpyDeviceToHost);
	if (cudaStatus_var10_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy var1 failed!");
		//free_arrays
		return;
	}

	//Copy output vector from GPU buffer to host memory.
	float* var40 = (float*)malloc(var40_num_of_bytes);
	cudaError_t cudaStatus_var40_memcpy = cudaMemcpy(var40, dev_var40, var40_num_of_bytes, cudaMemcpyDeviceToHost);
	if (cudaStatus_var40_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy var1 failed!");
		//free_arrays
		return;
	}

	// Copy output vector from GPU buffer to host memory.
	float2* var50 = (float2*)malloc(var50_num_of_bytes);
	cudaError_t cudaStatus_var50_memcpy = cudaMemcpy(var50, dev_var50, var50_num_of_bytes, cudaMemcpyDeviceToHost);
	if (cudaStatus_var50_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy var1 failed!");
		//free_arrays
		return;
	}

	// Copy output vector from GPU buffer to host memory.
	float* var60 = (float*)malloc(var60_num_of_bytes);
	cudaError_t cudaStatus_var60_memcpy = cudaMemcpy(var60, dev_var60, var60_num_of_bytes, cudaMemcpyDeviceToHost);
	if (cudaStatus_var60_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy var1 failed!");
		//free_arrays
		return;
	}

	// Copy output vector from GPU buffer to host memory.
	var70 = (float2*)malloc(var70_num_of_bytes);
	cudaError_t cudaStatus_var70_memcpy = cudaMemcpy(var70, dev_var70, var70_num_of_bytes, cudaMemcpyDeviceToHost);
	if (cudaStatus_var70_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy var1 failed!");
		//free_arrays
		return;
	}




	// Copy output vector from GPU buffer to host memory.
	var40_max_values = (float*)malloc(var40_max_values_num_of_bytes);
	cudaError_t cudaStatus_var40_max_values_memcpy = cudaMemcpy(var40_max_values, dev_var40_max_values, var40_max_values_num_of_bytes, cudaMemcpyDeviceToHost);
	if (cudaStatus_var40_max_values_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy var1 failed!");
		//free_arrays
		return;
	}

	// Copy output vector from GPU buffer to host memory.
	int* var40_indexes_of_max_values = (int*)malloc(var40_indexes_of_max_values_num_of_bytes);
	cudaError_t cudaStatus_var40_indexes_of_max_values_memcpy = cudaMemcpy(var40_indexes_of_max_values, dev_var40_indexes_of_max_values, var40_indexes_of_max_values_num_of_bytes, cudaMemcpyDeviceToHost);
	if (cudaStatus_var40_indexes_of_max_values_memcpy != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy var1 failed!");
		//free_arrays
		return;
	}


	write_data(folder_path, "var10.bin", var10_num_of_bytes, var10);
	write_data(folder_path, "var40.bin", var40_num_of_bytes, var40);
	write_data(folder_path, "var40_indexes_of_max_values.bin", var40_indexes_of_max_values_num_of_bytes, var40_indexes_of_max_values);
	write_data(folder_path, "var50.bin", var50_num_of_bytes, var50);
	write_data(folder_path, "var60.bin", var60_num_of_bytes, var60);

	cudaError_t cudaErr_var10 = cudaFree(dev_var10);
	if (cudaErr_var10 != cudaSuccess)
	{
		fprintf(stderr, "cudaFree(dev_var10) failed!");
	}
	cudaFree(dev_g1);
	cudaFree(dev_kk);
	cudaFree(dev_kk_size);
	cudaFree(dev_hh);
	cudaFree(dev_hh_size);
	cudaFree(dev_aa);
	cudaFree(dev_col_min);
	cudaFree(dev_row_min);
	cudaFree(dev_ww);
	cudaFree(dev_min_x);
	cudaFree(dev_min_y);
	cudaFree(dev_dim1);
	cudaFree(dev_dim2);
	cudaFree(dev_dim3);
	cudaFree(dev_cc);
	cudaFree(dev_dd);
	cudaFree(dev_ee);
	cudaFree(dev_ff);
	cudaFree(dev_similarity);
	cudaFree(dev_mm0);
	cudaFree(dev_mm1);
	cudaFree(dev_mm2);
	cudaFree(dev_g1dim1);
	cudaFree(dev_g2dim1);
	cudaFree(dev_var40);
	cudaFree(dev_var40_max_values);
	cudaFree(dev_var50);
	cudaFree(dev_var60);
	cudaFree(dev_var70);
}







//float2* build_tg1(float2* g1, float2 aa, vector<int2> mm, par_struct par)
//{
//	vector<int> tg1_rows;
//	vector<int> tg1_cols;
//	calc_tg1_rows_cols(aa, mm, par, tg1_rows, tg1_cols);
//	int tg1_rows_length = tg1_rows.size();
//	int tg1_cols_length = tg1_cols.size();
//	int num_of_tg1_elements = tg1_rows_length * tg1_cols_length;
//	float2* tg1 = new float2[num_of_tg1_elements];
//	for (int i = 0; i < tg1_rows_length; i++)
//	{
//		for (int j = 0; j < tg1_cols_length; j++)
//		{
//			int current_row_matlab = tg1_rows[i];
//			int current_col_matlab = tg1_cols[j];
//			int current_row = current_row_matlab - 1;
//			int current_col = current_col_matlab - 1;
//			int g1_current_index = current_col*g1dim1 + current_row;
//			int tg1_current_index = j * tg1_rows_length + i;
//			tg1[tg1_current_index] = g1[g1_current_index];
//		}
//	}
//	write_data(folder_path, "tg1.bin", sizeof(float2)*num_of_tg1_elements, tg1);
//	return tg1;	
//}




//float* build_var10(float2* var2, par_struct par, vector<int> dimensions, float2 aa, vector<int2> mm, float2* g1)
//{
//	
//	int dim1 = dimensions[0];
//	int dim2 = dimensions[1];
//	int dim3 = dimensions[2];
//	float* var10 = new float[dim1*dim2*dim3];
//	vector<int> tg1_rows;
//	vector<int> tg1_cols;
//	calc_tg1_rows_cols(aa, mm, par, tg1_rows, tg1_cols, dimensions);
//
//	for (int i = 0; i < dim1; i++)
//	{
//		for (int j = 0; j < dim2; j++)
//		{
//			float2 current_complex_val_tg1 = calc_tg1_val(i, j, g1, tg1_rows, tg1_cols);
//			for (int k = 0; k < dim3; k++)
//			{
//				int var10_current_index = k * dim1 * dim2 + j * dim1 + i;
//				int var2_current_index = var10_current_index;
//				int tg1_current_index = j * dim1 + i;
//				//float2 current_complex_val_tg1 = tg1[tg1_current_index];
//				float2 current_complex_val_var2 = var2[var2_current_index];
//				float2 diff_val;
//				diff_val.x = current_complex_val_tg1.x - current_complex_val_var2.x;
//				diff_val.y = current_complex_val_tg1.y - current_complex_val_var2.y;
//				float norm_diff = sqrt(diff_val.x * diff_val.x + diff_val.y * diff_val.y);
//				if (par.similarity == 1)
//				{
//					float norm_current_complex_val_tg1 = sqrt(current_complex_val_tg1.x * current_complex_val_tg1.x + current_complex_val_tg1.y * current_complex_val_tg1.y);
//					float norm_current_complex_val_var2 = sqrt(current_complex_val_var2.x*current_complex_val_var2.x + current_complex_val_var2.y*current_complex_val_var2.y);
//					float current_val = 0.5f * (norm_current_complex_val_tg1 + norm_current_complex_val_var2) - norm_diff;
//					var10[var10_current_index] = current_val;
//				}
//				else
//				{
//					float current_val = 1.0f - norm_diff;
//					var10[var10_current_index] = current_val;
//				}				
//			}
//		}
//	}
//	return var10;
//}

float* create_cumsum(float* arr, vector<int> dimensions, int dimension_for_cumsum)
{
	int dim1 = dimensions[0];
	int dim2 = dimensions[1];
	int dim3 = dimensions[2];
	float* cumsum_array = new float[dim1*dim2*dim3];
	for (int i = 0; i < dim1; i++)
	{
		for (int j = 0; j < dim2; j++)
		{
			for (int k = 0; k < dim3; k++)
			{
				int currentIndex = k * dim1 * dim2 + j * dim1 + i;
				if (dimension_for_cumsum == 0)
				{
					if (i == 0)
					{
						cumsum_array[currentIndex] = arr[currentIndex];
					}
					else
					{
						int prev_index_i = k * dim1 * dim2 + j * dim1 + (i-1);
						cumsum_array[currentIndex] = cumsum_array[prev_index_i] + arr[currentIndex];
					}
				}
				else if (dimension_for_cumsum == 1)
				{
					if (j == 0)
					{
						cumsum_array[currentIndex] = arr[currentIndex];
					}
					else
					{
						int prev_index_j = k * dim1 * dim2 + (j-1) * dim1 + i;
						cumsum_array[currentIndex] = cumsum_array[prev_index_j] + arr[currentIndex];
					}
				}
				else // (dimension_for_cumsum == 2)
				{
					if (k == 0)
					{
						cumsum_array[currentIndex] = arr[currentIndex];
					}
					else
					{
						int prev_index_k = (k-1) * dim1 * dim2 + j * dim1 + i;
						cumsum_array[currentIndex] = cumsum_array[prev_index_k] + arr[currentIndex];
					}
				}
			}
		}
	}
	return cumsum_array;
}

float* create_cumsum_dim1(float* arr, vector<int> dimensions)
{
	int dim1 = dimensions[0];
	int dim2 = dimensions[1];
	int dim3 = dimensions[2];
	float* cumsum_array = new float[dim1*dim2*dim3];
	for (int j = 0; j < dim2; j++)
	{
		for (int k = 0; k < dim3; k++)
		{
			for (int i = 0; i < dim1; i++)
			{
				int currentIndex = k * dim1 * dim2 + j * dim1 + i;
				if (i == 0)
				{
					cumsum_array[currentIndex] = arr[currentIndex];
				}
				else
				{
					int prev_index_i = k * dim1 * dim2 + j * dim1 + (i - 1);
					cumsum_array[currentIndex] = cumsum_array[prev_index_i] + arr[currentIndex];
				}
			}
		}
	}
	return cumsum_array;
}

float* create_cumsum_dim2(float* arr, vector<int> dimensions)
{
	int dim1 = dimensions[0];
	int dim2 = dimensions[1];
	int dim3 = dimensions[2];
	float* cumsum_array = new float[dim1*dim2*dim3];
	for (int i = 0; i < dim1; i++)
	{
		for (int k = 0; k < dim3; k++)
		{
			for (int j = 0; j < dim2; j++)
			{
				int currentIndex = k * dim1 * dim2 + j * dim1 + i;
				if (j == 0)
				{
					cumsum_array[currentIndex] = arr[currentIndex];
				}
				else
				{
					int prev_index_j = k * dim1 * dim2 + (j - 1) * dim1 + i;
					cumsum_array[currentIndex] = cumsum_array[prev_index_j] + arr[currentIndex];
				}
			}
		}
	}
	return cumsum_array;
}

void read_data(string filename, int num_of_bytes, float2* var)
{
	char* var_bytes = (char*)var;
	const char* str = filename.c_str();
	std::ifstream var1File(filename, std::ios::in | std::ios::binary);
	if (var1File.is_open())
	{
		var1File.read(var_bytes, num_of_bytes);
	}
	else
	{
		fprintf(stderr, "Could not open file ..\\..\\..\\bin_files\\g1.bin");
		getchar();
	}
}

//float calc_var40_val(int col_min, int row_min, int ww, int min_x, int min_y, vector<int> dimensions, float2* g1, float2 aa, vector<int2> mm, par_struct par, int i, int j, int k, int ax, int ay, int axy, int cols_step, int rows_step)
//{
//	float current_val1;
//	float current_val2;
//	float current_val3;
//	float current_val4;
//
//
//	current_val1 = calc_var90_val(i, j, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//	current_val2 = calc_var90_val(i + ay, j + ax, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//	current_val3 = calc_var90_val(i, j + ax, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//	current_val4 = calc_var90_val(i + ay, j, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//
//	for (int w = i - 1; w >= 0; w--)
//	{
//		current_val1 = current_val1 + calc_var90_val(w, j, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//		current_val3 = current_val3 + calc_var90_val(w, j + ax, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//	}
//
//	for (int w = i + ay - 1; w >= 0; w--)
//	{
//		current_val2 = current_val2 + calc_var90_val(w, j + ax, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//		current_val4 = current_val4 + calc_var90_val(w, j, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//	}
//
//	for (int w = j - 1; w >= 0; w--)
//	{
//		current_val1 = current_val1 + calc_var90_val(i, w, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//		current_val4 = current_val4 + calc_var90_val(i + ay, w, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//	}
//
//	for (int w = j + ax - 1; w >= 0; w--)
//	{
//		current_val2 = current_val2 + calc_var90_val(i + ay, w, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//		current_val3 = current_val3 + calc_var90_val(i, w, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//	}
//
//	float var40_val = (current_val1 + current_val2 - current_val3 - current_val4) / axy;
//	return var40_val;
//}

//float* build_var91(int col_min, int row_min, int ww, int min_x, int min_y, vector<int> dimensions, int jj, int rr, float2* g1, float2 aa, vector<int2> mm, par_struct par, vector<int>& var40_dimensions, vector<int> var30_dimensions)
//{
//	int cols_step = jj;
//	int rows_step = rr;
//	int nr = (mm[2].y - mm[1].y + 1) / par.cc;
//	int nc = (mm[1].x - mm[0].x + 1) / par.dd;
//	var40_dimensions.push_back(nr);
//	var40_dimensions.push_back(nc);
//	int dim3 = par.hh.size() * par.kk.size();
//	var40_dimensions.push_back(dim3);
//
//	int ax = 2 * par.ee / par.dd;
//	int ay = 2 * par.ff / par.cc;
//	int axy = ax * ay;
//
//	float* var40 = new float[nr*nc*dim3];
//
//	for (int i = 0; i < nr; i++)
//	{
//		std::cout << "var40: Iteration i = " << i+1 << " out of " << nr << " iterations." << std::endl;
//		for (int j = 0; j < nc; j++)
//		{
//			float2 current_complex_val_tg1_1 = calc_tg1_val(aa, mm, par, i, j, g1);
//			float2 current_complex_val_tg1_2 = calc_tg1_val(aa, mm, par, i + ay, j + ax, g1);
//			float2 current_complex_val_tg1_3 = calc_tg1_val(aa, mm, par, i, j + ax, g1);
//			float2 current_complex_val_tg1_4 = calc_tg1_val(aa, mm, par, i + ay, j, g1);
//			for (int k = 0; k < dim3; k++)
//			{				
//				int index1 = k * nr * nc + j * nr + i;
//				float currentVal = calc_var40_val(col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, i, j, k, ax, ay, axy, cols_step, rows_step);
//				var40[index1] = currentVal;
//			}
//		}
//	}
//	return var40;
//}


//void get_max_val_and_index(float& max_val, int& index_of_max_val, float2 aa, vector<int2> mm, par_struct par, float2* g1, int col_min, int row_min, int ww, int min_x, int min_y, vector<int> dimensions, int cols_step, int rows_step, int i, int j)
//{
//	int ax = 2 * par.ee / par.dd;
//	int ay = 2 * par.ff / par.cc;
//	int axy = ax * ay;
//	int dim3 = par.hh.size() * par.kk.size();
//
//	//float2 current_complex_val_tg1_1 = calc_tg1_val(aa, mm, par, i, j, g1);
//	//float2 current_complex_val_tg1_2 = calc_tg1_val(aa, mm, par, i + ay, j + ax, g1);
//	//float2 current_complex_val_tg1_3 = calc_tg1_val(aa, mm, par, i, j + ax, g1);
//	//float2 current_complex_val_tg1_4 = calc_tg1_val(aa, mm, par, i + ay, j, g1);
//	for (int k = 0; k < dim3; k++)
//	{
//		float current_val1 = calc_var90_val(i, j, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//		float current_val2 = calc_var90_val(i + ay, j + ax, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//		float current_val3 = calc_var90_val(i, j + ax, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//		float current_val4 = calc_var90_val(i + ay, j, k, col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, cols_step, rows_step);
//		float currentVal = (current_val1 + current_val2 - current_val3 - current_val4) / axy;
//		if (k == 0 || currentVal > max_val)
//		{
//			max_val = currentVal;
//			index_of_max_val = k;
//		}
//	}
//}

float2* build_var50(par_struct par, int* matrix_indexes_of_max_values, vector<int> dimensions)
{
	int search_dim1 = KK_SIZE;
	int search_dim2 = HH_SIZE;

	float* xsearch = new float[search_dim1*search_dim2];
	float* ysearch = new float[search_dim1*search_dim2];

	for (int i = 0; i < search_dim1; i++)
	{
		for (int j = 0; j < search_dim2; j++)
		{
			int current_index = j * search_dim1 + i;
			xsearch[current_index] = par.hh[j];
			ysearch[current_index] = par.kk[i];
		}
	}

	float2* var50 = new float2[dimensions[0] * dimensions[1]];

	for (int i = 0; i < dimensions[0]; i++)
	{
		for (int j = 0; j < dimensions[1]; j++)
		{
			int current_index = j * dimensions[0] + i;
			int current_index_for_xysearch_matlab = matrix_indexes_of_max_values[current_index];
			int current_index_for_xysearch = current_index_for_xysearch_matlab - 1;
			float xsearch_val = xsearch[current_index_for_xysearch];
			float ysearch_val = ysearch[current_index_for_xysearch];
			float2 current_val;
			current_val.x = xsearch_val;
			current_val.y = ysearch_val;
			var50[current_index] = current_val;
		}
	}

	delete[] xsearch;
	delete[] ysearch;
	return var50;
}



//float2* build_var92(int col_min, int row_min, int ww, int min_x, int min_y, vector<int> dimensions, int jj, int rr, float2* g1, float2 aa, vector<int2> mm, par_struct par, vector<int>& var40_dimensions, vector<int> var30_dimensions)
//{
//	int cols_step = jj;
//	int rows_step = rr;
//	int nr = (mm[2].y - mm[1].y + 1) / par.cc;
//	int nc = (mm[1].x - mm[0].x + 1) / par.dd;
//	int dim3 = par.hh.size() * par.kk.size();
//	if (var40_dimensions.size() == 0)
//	{
//		var40_dimensions.push_back(nr);
//		var40_dimensions.push_back(nc);
//		var40_dimensions.push_back(dim3);
//	}
//
//	//vector<int> rs;
//	//for (int i = 1; i <= nr; i++)
//	//{
//	//	rs.push_back(i);
//	//}
//
//	//vector<int> cs;
//	//for (int i = 1; i <= nc; i++)
//	//{
//	//	cs.push_back(i);
//	//}
//
//	int search_dim1 = par.kk.size();
//	int search_dim2 = par.hh.size();
//
//	float* xsearch = new float[search_dim1*search_dim2];
//	float* ysearch = new float[search_dim1*search_dim2];
//
//	for (int i = 0; i < search_dim1; i++)
//	{
//		for (int j = 0; j < search_dim2; j++)
//		{
//			int current_index = j * search_dim1 + i;
//			xsearch[current_index] = par.hh[j];
//			ysearch[current_index] = par.kk[i];
//		}
//	}
//
//
//
//	float2* var50 = new float2[nr*nc];
//	float max_val;
//	int current_index_for_xysearch_matlab;
//
//	for (int i = 0; i < nr; i++)
//	{
//		for (int j = 0; j < nc; j++)
//		{
//			get_max_val_and_index(max_val, current_index_for_xysearch_matlab,aa, mm, par, g1, col_min, row_min, ww, min_x, min_y, dimensions, cols_step, rows_step, i, j);
//
//			int current_index = j * nr + i;
//			int current_index_for_xysearch = current_index_for_xysearch_matlab - 1;
//			float xsearch_val = xsearch[current_index_for_xysearch];
//			float ysearch_val = ysearch[current_index_for_xysearch];
//			float2 current_val;
//			current_val.x = xsearch_val;
//			current_val.y = ysearch_val;
//			var50[current_index] = current_val;
//		}
//	}
//
//	delete[] xsearch;
//	delete[] ysearch;
//	return var50;
//}


//float* build_var40(vector<int2> mm, par_struct par, float* var30, int var30_dimension3, vector<int> var30_dimensions, vector<int>& var40_dimensions)
//{
//	int nr = (mm[2].y - mm[1].y + 1) / par.cc;
//	int nc = (mm[1].x - mm[0].x + 1) / par.dd;
//	var40_dimensions.push_back(nr);
//	var40_dimensions.push_back(nc);
//	var40_dimensions.push_back(var30_dimension3);
//
//	int ax = 2 * par.ee / par.dd;
//	int ay = 2 * par.ff / par.cc;
//	int axy = ax * ay;
//
//	float* var40 = new float[nr*nc*var30_dimension3];
//
//	for (int i = 0; i < nr; i++)
//	{
//		for (int j = 0; j < nc; j++)
//		{
//			for (int k = 0; k < var30_dimension3; k++)
//			{
//				int index1 = k * nr * nc + j * nr + i; 
//				int index2 = k * var30_dimensions[0] * var30_dimensions[1] + j * var30_dimensions[0] + i;
//				int index3 = k * var30_dimensions[0] * var30_dimensions[1] + (j + ax) * var30_dimensions[0] + (i + ay);
//				int index4 = k * var30_dimensions[0] * var30_dimensions[1] + (j + ax) * var30_dimensions[0] + i;
//				int index5 = k * var30_dimensions[0] * var30_dimensions[1] + j * var30_dimensions[0] + (i + ay);
//				float currentVal = (var30[index2] + var30[index3] - var30[index4] - var30[index5]) / axy;
//				var40[index1] = currentVal;
//			}
//		}
//	}
//	return var40;
//}

void calc_max_in_third_dimension(float* matrix, vector<int> dimensions, float* &matrix_max_values, int* &matrix_indexes_of_max_values)
{
	int dim1 = dimensions[0];
	int dim2 = dimensions[1];
	int dim3 = dimensions[2];
	matrix_max_values = new float[dim1 * dim2];
	matrix_indexes_of_max_values = new int[dim1 * dim2];
	for (int i = 0; i < dim1; i++)
	{
		for (int j = 0; j < dim2; j++)
		{
			for (int k = 0; k < dim3; k++)
			{
				int current_index_for_3D_matrix = k * dim1 * dim2 + j * dim1 + i;
				int current_index_for_2D_matrix = j * dim1 + i;
				float current_val = matrix[current_index_for_3D_matrix];
				if (k == 0 || current_val > matrix_max_values[current_index_for_2D_matrix])
				{
					matrix_max_values[current_index_for_2D_matrix] = current_val;
					matrix_indexes_of_max_values[current_index_for_2D_matrix] = k;
				}
			}
		}
	}
}



//float* build_var93(int col_min, int row_min, int ww, int min_x, int min_y, vector<int> dimensions, int jj, int rr, float2* g1, float2 aa, vector<int2> mm, par_struct par, vector<int>& var40_dimensions, vector<int> var30_dimensions, float* var30)
//{
//	int cols_step = jj;
//	int rows_step = rr;
//	int nr = (mm[2].y - mm[1].y + 1) / par.cc;
//	int nc = (mm[1].x - mm[0].x + 1) / par.dd;
//	var40_dimensions.push_back(nr);
//	var40_dimensions.push_back(nc);
//	int dim3 = par.hh.size() * par.kk.size();
//	var40_dimensions.push_back(dim3);
//
//	int ax = 2 * par.ee / par.dd;
//	int ay = 2 * par.ff / par.cc;
//	int axy = ax * ay;
//
//	float* var40 = new float[nr*nc*dim3];
//
//	for (int i = 0; i < nr; i++)
//	{
//		for (int j = 0; j < nc; j++)
//		{
//			float2 current_complex_val_tg1_1 = calc_tg1_val(aa, mm, par, i, j, g1);
//			float2 current_complex_val_tg1_2 = calc_tg1_val(aa, mm, par, i + ay, j + ax, g1);
//			float2 current_complex_val_tg1_3 = calc_tg1_val(aa, mm, par, i, j + ax, g1);
//			float2 current_complex_val_tg1_4 = calc_tg1_val(aa, mm, par, i + ay, j, g1);
//			for (int k = 0; k < dim3; k++)
//			{
//				int index1 = k * nr * nc + j * nr + i;
//				float var40_val = calc_var40_val(col_min, row_min, ww, min_x, min_y, dimensions, g1, aa, mm, par, i, j, k, current_complex_val_tg1_1, current_complex_val_tg1_2, current_complex_val_tg1_3, current_complex_val_tg1_4, ax, ay, axy, cols_step, rows_step);
//				var40[index1] = currentVal;
//			}
//		}
//	}
//	return var40;
//}



float* build_var60(par_struct par, float* var40, int* var40_indexes_of_max_values, vector<int> var40_dimensions)
{
	int nr = var40_dimensions[0];
	int nc = var40_dimensions[1];
	int var40_dim3 = var40_dimensions[2];
	int nsearchy = KK_SIZE;
	bool is1dsearch = nsearchy == 1;
	int length_small_arr = 3; //length of the array {-1, 0, 1} or { -nsearchy, 0, nsearchy }
	float very_small_number = -999999;

	int di_length;
	if (is1dsearch)
	{
		di_length = length_small_arr;
	}
	else
	{
		di_length = length_small_arr * length_small_arr;
	}
	
	float* var60 = new float[nr*nc*di_length];
	var40[0] = very_small_number;

	int var40_mult_dimensions = var40_dimensions[0] * var40_dimensions[1] * var40_dimensions[2];
	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			for (int k = 0; k < di_length; k++)
			{
				int di_val;
				if (is1dsearch)
				{
					di_val = k - 1; //values - 1, 0, 1
				}
				else
				{
					int arr1_index = k / length_small_arr; // indexes 0, 1, 2
					int arr1_val = arr1_index - 1; //values - 1, 0, 1
					int arr2_index = k % length_small_arr; //indexes 0, 1, 2
					int arr2_val = (arr2_index - 1) * nsearchy; //values - 5, 0, 5
					di_val = arr1_val + arr2_val;
				}
					
				int current_index_var40_indexes = j*nr + i;
				int var40_current_index = var40_indexes_of_max_values[current_index_var40_indexes];
				int var40_current_index_matlab = var40_current_index + 1;
				int current_val = var40_current_index_matlab + di_val;
				int val1 = (current_val - 1)*nr*nc;
				int current_index_for_xx_yy = j*nr + i;

				int xx_val = (current_index_for_xx_yy / nr) + 1;
				int yy_val = (current_index_for_xx_yy % nr) + 1;

				int val2 = (xx_val - 1)*nr;
				int intermediate_lind3_val = val1 + val2 + yy_val;
				int lind3_val_matlab;
				if (intermediate_lind3_val <= 0 || intermediate_lind3_val > var40_mult_dimensions)
				{
					lind3_val_matlab = 1;
				}
				else
				{
					lind3_val_matlab = intermediate_lind3_val;
				}

				int lind3_val = lind3_val_matlab - 1;
				int current_index_for_var60 = k*nr*nc + j*nr + i;
				float current_var60_val = var40[lind3_val];
				if (i == (nr - 1) && j == (nc - 1) && k == (di_length - 1))
				{
					int david = 5;
				}
				var60[current_index_for_var60] = current_var60_val;					
			}
		}
	}

	return var60;	
}

float calc_sft_val_from_vector(vector<float> mean_vector)
{
	float a = mean_vector[0];
	float b = mean_vector[1];
	float c = mean_vector[2];
	float denominator = a + c - 2 * b;
	float eps = 0.0000001;
	float sft_val;
	if (denominator < eps)
	{
		sft_val = 0;
	}
	else
	{
		sft_val = ((a - c) / 2) / denominator;
	}
	

	if (sft_val < -0.5)
	{
		sft_val = -0.5;
	}
		
	if (sft_val > 0.5)
	{
		sft_val = 0.5;
	}

	return sft_val;
}

float calc_sft_val_from_matrix(float* matrix, int i, int j, int dim1, int dim2, int dim)
{
	int dim3 = 3;
	int dim4 = 3;
	float sft_val = 0;
	if (dim == 4)
	{
		vector<float> mean_vector;
		for (int k = 0; k < dim3; k++)
		{
			float current_sum = 0;
			for (int w = 0; w < dim4; w++)
			{
				int current_index = w*dim3*dim2*dim1 + k*dim2*dim1 + j*dim1 + i;
				current_sum = current_sum + matrix[current_index];
			}
			float current_mean = current_sum / dim4;
			mean_vector.push_back(current_mean);
		}
		sft_val = calc_sft_val_from_vector(mean_vector);
	}

	if (dim == 3)
	{
		vector<float> mean_vector;
		for (int w = 0; w < dim4; w++)
		{
			float current_sum = 0;
			for (int k = 0; k < dim3; k++)
			{
				int current_index = w * dim3*dim2*dim1 + k * dim2*dim1 + j * dim1 + i;
				current_sum = current_sum + matrix[current_index];
			}
			float current_mean = current_sum / dim3;
			mean_vector.push_back(current_mean);
		}
		sft_val = calc_sft_val_from_vector(mean_vector);
	}

	return sft_val;
}

float2* build_var70(par_struct par, float* var60, vector<int> dimensions)
{
	int nr = dimensions[0];
	int nc = dimensions[1];
	int dsearchx = abs(par.hh[1] - par.hh[0]);
	int dsearchy = abs(par.kk[1] - par.kk[0]);
	int nsearchy = KK_SIZE;
	bool is1dsearch = nsearchy == 1;
	float2* var70 = new float2[nr*nc];
	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			int current_index = j*nr + i;
			if (is1dsearch == false)
			{
				float sftx_current_val = calc_sft_val_from_matrix(var60, i, j, nr, nc, 4);
				float sfty_current_val = calc_sft_val_from_matrix(var60, i, j, nr, nc, 3);
				float2 var70_current_val;
				var70_current_val.x = sftx_current_val * dsearchx;
				var70_current_val.y = sfty_current_val * dsearchy;
				var70[current_index] = var70_current_val;
			}
			else
			{
				int index1 = 0 * nr * nc + j * nr + i;
				int index2 = 1 * nr * nc + j * nr + i;
				int index3 = 2 * nr * nc + j * nr + i;
				float a = var60[index1];
				float b = var60[index2];
				float c = var60[index3];
				vector<float> vec = { a,b,c };
				float sftx_current_val = calc_sft_val_from_vector(vec);
				float2 var70_current_val;
				var70_current_val.x = sftx_current_val * dsearchx;
				var70_current_val.y = 0;
				var70[current_index] = var70_current_val;
			}
		}
	}
	return var70;
}

float2* build_var80(float2* var50, float2* var70, vector<int> dimensions)
{
	int nr = dimensions[0];
	int nc = dimensions[1];
	float2* var80 = new float2[nr*nc];

	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			int current_index = j * nr + i;
			var80[current_index].x = var50[current_index].x + var70[current_index].x;
			var80[current_index].y = var50[current_index].y + var70[current_index].y;
		}
	}
	return var80;
}

int get_min_element(int* arr, int length)
{
	if (length <= 0)
	{
		throw "no values in array";
	}
	int min_val = arr[0];
	for (int i = 1; i < length; i++)
	{
		if (arr[i] < min_val)
		{
			min_val = arr[i];
		}
	}
	return min_val;

}

void try_cuda(float2* g1, float2* g2, float2 aa, float2 bb, vector<int2> mm, par_struct par, int iblk, float*& var40_max_values, float2*& var70)
{
	int pp = HH_SIZE;
	int ww = KK_SIZE;
	int min_x = get_min_element(par.hh, HH_SIZE);
	int min_y = get_min_element(par.kk, KK_SIZE);
	int col_min = mm[0].x - par.ee + bb.x;
	int row_min = mm[1].y - par.ff + bb.y;
	int dimension1 = floor((mm[2].y + 2 * par.ff - mm[1].y) / par.cc) + 1;
	int dimension2 = floor((mm[1].x + 2 * par.ee - mm[0].x) / par.dd) + 1;
	int dimension3 = HH_SIZE * KK_SIZE;
	vector<int> dimensions = { dimension1, dimension2, dimension3 };
	int cc = par.cc;
	int dd = par.dd;
	int ee = par.ee;
	int ff = par.ff;
	int similarity = par.similarity;
	int* kk = par.kk;
	int* hh = par.hh;

	build_var90_before_cuda(col_min, row_min, ww, min_x, min_y, dimensions, g1, g2, aa, mm, cc, dd, ee, ff, similarity, kk, hh, var40_max_values, var70); //should be the same as var10

	int nr = (mm[2].y - mm[1].y + 1) / par.cc;
	int nc = (mm[1].x - mm[0].x + 1) / par.dd;
	write_data(folder_path, "var40_max_values.bin", sizeof(float)*nr * nc, var40_max_values);
	write_data(folder_path, "var70.bin", sizeof(float2)*nr * nc, var70);
}


void create_folder(string folder_path)
{
	if (!CreateDirectory(folder_path.c_str(), NULL))
	{
		return;
	}
}

bool folderExists(const std::string& dirName_in)
{
	DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
	if (ftyp == INVALID_FILE_ATTRIBUTES)
		return false;  //something is wrong with your path!

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		return true;   // this is a directory!

	return false;    // this is not a directory!
}

int main()
{
	float2* g1;
	float2* g2;
	float2 aa;
	float2 bb;
	vector<int2> mm;
	par_struct par;
	int iblk;
	
	if (folderExists(folder_path) == false)
		create_folder(folder_path);


	g1 = build_random_float2_matrix(g1dim1, g1dim2);
	g2 = build_random_float2_matrix(g2dim1, g2dim2);

	write_data(folder_path, "g1.bin", sizeof(float2)*g1dim1*g1dim2, g1);
	write_data(folder_path, "g2.bin", sizeof(float2)*g2dim1*g2dim2, g2);


	load_variables(g1, g2, aa, bb, mm, par, iblk);

	float* var40_max_values;
	float2* var70;
	try_cuda(g1, g2, aa, bb, mm, par, iblk, var40_max_values, var70);


	delete[] var70;
	delete[] var40_max_values;
	return 0;
}