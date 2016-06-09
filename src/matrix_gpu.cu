#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_gpu.h"

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
		float demominator = in[i*size + i];
		divide_row_gpu<<<size/32 + 1,32>>>(demominator, d_out, i*size, size);
		divide_row_gpu<<<size/32 + 1,32>>>(demominator, d_in, i*size, size);

		gpuErrchk(cudaMemcpy(out + i*size, d_out + i*size, size*sizeof(float), cudaMemcpyDeviceToHost))
		gpuErrchk(cudaMemcpy(in + i*size, d_in + i*size, size*sizeof(float), cudaMemcpyDeviceToHost))

		//zero out the column below
		int j;
		for(j = i + 1; j < size; j++){

			float scale = in[j*size + i];
			subtract_row_gpu<<<size/32 + 1, 32>>>(d_out + i*size, d_out + j*size, scale, size);
			subtract_row_gpu<<<size/32 + 1, 32>>>(d_in + i*size, d_in + j*size, scale, size); // in row, out/target row, scale the in row, size
			gpuErrchk(cudaMemcpy(out + j*size, d_out + j*size, size*sizeof(float), cudaMemcpyDeviceToHost))
			gpuErrchk(cudaMemcpy(in + j*size, d_in + j*size, size*sizeof(float), cudaMemcpyDeviceToHost))
		}
	}


	//back substitution step
	int column;
	for(column = size - 1; column >= 1; column--){
		int row;
		for(row = column-1; row >= 0; row--){
			float factor = in[row*size + column];

			int j;
			for(j = 0; j < size; j++){
				out[row*size + j] = out[row*size + j] - factor*out[column*size +j];
				in[row*size + j] = in[row*size + j] - factor*in[column*size +j];
			}
		}
	}

	*success = 1;
}

// in row, out/target row, scale the in row, size
__global__
void subtract_row_gpu(float * source, float * target, float scale, int size){
	int idx = blockIdx.x*blockDim.x  + threadIdx.x;
	if(idx < size){
		target[idx] = target[idx] - (source[idx] * scale);
	}
}

__global__
void divide_row_gpu(float denominator,float * vector, int start_idx, int size){
	int idx = blockIdx.x*blockDim.x  + threadIdx.x;
	if(idx < size){
		vector[idx + start_idx] = vector[idx + start_idx]/denominator;
	}
}
