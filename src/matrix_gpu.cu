#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_gpu.h"
#include "testing_util.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void inverse_gpu(float * in, int size, float * out, int * success){

	float * d_in;
	float * d_out;
	gpuErrchk(cudaMalloc((void **)&d_in, size*size* sizeof(float)))
	gpuErrchk(cudaMalloc((void **)&d_out, size*size* sizeof(float)))

	gpuErrchk(cudaMemcpy(d_in, in, size*size*sizeof(float), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(d_out, out, size*size*sizeof(float), cudaMemcpyHostToDevice))

	//Gaussian elimination step
	int i;
	for(i = 0; i < size; i++){
		//todo, there was a check for the possibility to invert here before (row swap), should be brought back adventually

		//scale the row so that [i][i] == 1
		divide_2rows_gpu<<<size/32 + 1,32>>>(i*size + i,d_in , d_in + i*size, d_out + i*size, size);
		//zero out the column below
		zero_out_column_gpu<<<size/32 + 1,32>>>(i, 1, d_in, d_out, size);
	}

	//back substitution step
	int column;
	for(column = size - 1; column >= 1; column--){
		zero_out_column_gpu<<<size/32 + 1,32>>>(column, -1, d_in, d_out, size);
		print_matrix_on_device_kernel<<<1,1>>>(d_in, size);
	}

	//get the inverted matrix back to host memory
	gpuErrchk(cudaMemcpy(out, d_out, size*size*sizeof(float), cudaMemcpyDeviceToHost))

	*success = 1;
}


__global__
void zero_out_column_gpu(int column, int direction, float * in, float * out, int size){
	int idx = blockIdx.x*blockDim.x  + threadIdx.x;
	if(idx < size){
		int j;
		for(j = column + direction; j < size && j >= 0; j+= direction){

			float scale = in[j*size + column];
			out[idx + j*size] = out[idx+j*size] - (out[idx + column*size] * scale);
			in[idx + j*size] = in[idx+j*size] - (in[idx + column*size] * scale);
		}
	}
}

__global__
void subtract_rows_gpu(int i, float * in, float * out, int size){
	int idx = blockIdx.x*blockDim.x  + threadIdx.x;
	if(idx < size){
		int j;
		for(j = i + 1; j < size; j++){

			float scale = in[j*size + i];
			out[idx + j*size] = out[idx+j*size] - (out[idx + i*size] * scale);
			in[idx + j*size] = in[idx+j*size] - (in[idx + i*size] * scale);
		}
	}
}


//takes vector[denominator_idx] as index and divides all elements in the row from vector and vector2
__global__
void divide_2rows_gpu(int denominator_idx, float * denom_src_vec, float * vector, float * vector2, int size){
	int idx = blockIdx.x*blockDim.x  + threadIdx.x;
	float denominator = denom_src_vec[denominator_idx];

	__syncthreads();
	if(idx < size){
		vector[idx] = vector[idx]/denominator;
		vector2[idx] = vector2[idx]/denominator;
	}
}

__global__
void subtract_row_gpu(float * source, float * target, float scale, int size){
	int idx = blockIdx.x*blockDim.x  + threadIdx.x;
	if(idx < size){
		target[idx] = target[idx] - (source[idx] * scale);
	}
}

__global__
void divide_row_gpu(int denominator_idx, float * denom_src_vec, float * vector, int start_idx, int size){
	int idx = blockIdx.x*blockDim.x  + threadIdx.x;
	float denominator = denom_src_vec[denominator_idx];
	__syncthreads();
	if(idx < size){
		vector[idx + start_idx] = vector[idx + start_idx]/denominator;
	}
}
