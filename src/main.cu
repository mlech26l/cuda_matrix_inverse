// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>


#include "random_matrix.h"
#include "unity_matrix.h"
#include "matrix_multiplication.h"
#include "matrix.h"



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Function for testing the matrix multiplication and check for unity matrix
void test_matrix_util_functions(void);

int main(int argc, char **argv)
{
	test_matrix_util_functions();
	exit(EXIT_SUCCESS);
}


__global__ void reduce0(float *_idata, float *_odata, int size) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	for(unsigned int s=1; s < size; s *= 2) {
		if(i*s < size){
			_idata[i] += _idata[i*2*s];
		}
		__syncthreads();
	}

	if (i == 0) {
		_odata[0] = _idata[i];
	}
}

float * create_identity_matrix(int size){
	float * out = (float *)malloc(sizeof(float)*size*size);
	int i;
	for(i = 0; i < size; i++){
		int j;
		for(j = 0;j < size; j++){
			out[i*size + j] = 0.0;
		}
		out[i*size + i] = 1.0;
	}
	return out;
}

int main1(int argc, char **argv)
{
	float * matrix;


	float * ret  = 0;

	int size = 2;

	matrix = create_identity_matrix(size);
	print_matrix(matrix,size);

	int total_size = size*size;

	float *d_matrix, *d_answer;
	int * d_size;

	//malloc section
	gpuErrchk(cudaMalloc((void **)&d_matrix, total_size * sizeof(float)))

	gpuErrchk(cudaMalloc((void **)&d_answer, 1 * sizeof(float)))

	gpuErrchk(cudaMalloc((void **)&d_size, 1 * sizeof(int)))

/////////////////////

	///data transfer
	gpuErrchk(cudaMemcpy(d_matrix, matrix, total_size * sizeof(float), cudaMemcpyHostToDevice))

	gpuErrchk(cudaMemcpy(d_size, &total_size, 1 * sizeof(int), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(d_answer, ret, 1 * sizeof(float), cudaMemcpyHostToDevice))

/////////////////////

	int threadsPerBlock = 1;
	int blocksPerGrid = 1;

	reduce0<<<blocksPerGrid, threadsPerBlock>>>(d_matrix,d_answer,total_size);

	gpuErrchk(cudaMemcpy(ret, d_answer, 1 * sizeof(float), cudaMemcpyDeviceToHost));
	printf("sum of the matrix is = %f\n",*ret);

	free(matrix);
	return 0;
}


void test_matrix_util_functions(void)
{
	float *h_mat, *d_mat;
	int n = 12;
	
	/* Allocate n floats on host */
	h_mat = (float *)malloc(n*n* sizeof(float));
	/* Allocate n floats on device */

	d_mat = generate_random_matrix(n,100,1);
	
	/* Copy device memory to host */
	if(cudaMemcpy(h_mat, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToHost)!= cudaSuccess)
	{
		printf("Error at cudaMalloc! ");
		exit(EXIT_FAILURE);
	}

	
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_mat[x*n + y]);
		}
		printf("\n");
	} 
	

	float *d_b, *d_c;
	if(cudaMalloc((void **)&d_b, n*n* sizeof(float)) != cudaSuccess)
	{
		printf("Error on Cuda Malloc!\n");
		exit(EXIT_FAILURE);
	}
	if(cudaMemcpy(d_b, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToDevice)!= cudaSuccess)
	{
		printf("Error at cudaMalloc! ");
		exit(EXIT_FAILURE);
	}
	
	if(cudaMalloc((void **)&d_c, n*n*sizeof(float)) != cudaSuccess)
	{
		printf("Error on Cuda Malloc!\n");
		exit(EXIT_FAILURE);
	}
	
	matrix_multiply(d_c,d_mat,d_mat,n);
	
	
	if(cudaMemcpy(h_mat, d_c, n*n * sizeof(float), cudaMemcpyDeviceToHost)!= cudaSuccess)
	{
		printf("Error at cudaMalloc! ");
		exit(EXIT_FAILURE);
	}
	printf("Squared matrix:\n");
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_mat[x*n + y]);
		}
		printf("\n");
	} 
	
	
	float* d_unity=get_dev_unity_matrix(n);
	matrix_multiply(d_c,d_mat,d_unity,n);
	
	if(cudaMemcpy(h_mat, d_unity, n*n * sizeof(float), cudaMemcpyDeviceToHost)!= cudaSuccess)
	{
		printf("Error at cudaMalloc! ");
		exit(EXIT_FAILURE);
	}
	printf("unity matrix:\n");
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_mat[x*n + y]);
		}
		printf("\n");
	} 
	
	if(cudaMemcpy(h_mat, d_b, n*n * sizeof(float), cudaMemcpyDeviceToHost)!= cudaSuccess)
	{
		printf("Error at cudaMemcpy! ");
		exit(EXIT_FAILURE);
	}
	printf("Random times unity matrix:\n");
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_mat[x*n + y]);
		}
		printf("\n");
	} 
	int ur = is_unity_matrix(d_mat,n);
	printf("Is random matrix unit: %d\n",ur);
	ur = is_unity_matrix(d_unity,n);
	printf("Is unit matrix unit: %d\n",ur);
	
	free(h_mat);
	cudaFree(d_mat);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_unity);
}
