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
#include "lup_decomposition.h"



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
	int n = 3;
	
	/* Allocate n floats on host */
	h_mat = (float *)malloc(n*n* sizeof(float));
	/* Allocate n floats on device */

	d_mat = generate_random_matrix(n,100,1);
	
	/* Copy random matrix to host */
	if(cudaMemcpy(h_mat, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToHost)!= cudaSuccess)
	{
		printf("Error at cudaMalloc! ");
		exit(EXIT_FAILURE);
	}

	/* Print out random generated matrix */
	printf("Random generated Matrix:\n");
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_mat[x*n + y]);
		}
		printf("\n");
	} 
	printf("WA output form:\n");
	printf("inverse {");
	for(int x = 0; x < n; x++) {
		printf("{");
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_mat[x*n + y]);
			if(y != n-1)
				printf(", ");
		}
		printf("}");
		if(x != n-1)
			printf(", ");
	} 
	printf("}\n");
	
	/* Invert matrix on host using LU decomposition */
	if(matrix_inverse_host_lup(h_mat,n) < 0)
	{
		printf("Matrix Singular!\n");
		exit(EXIT_SUCCESS);
	}
	
	/* Print out inverse matrix */
	printf("Inverse Matrix:\n");
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_mat[x*n + y]);
		}
		printf("\n");
	} 
	
	/* Allocate second matrix on device for multiplication */
	float *d_inv;
	if(cudaMalloc((void **)&d_inv, n*n* sizeof(float)) != cudaSuccess)
	{
		printf("Error on Cuda Malloc!\n");
		exit(EXIT_FAILURE);
	}

	/* Copy inverse matrix to device */
	if(cudaMemcpy(d_inv, h_mat, n*n * sizeof(float), cudaMemcpyHostToDevice)!= cudaSuccess)
	{
		printf("Error at cudaMemcpy! ");
		exit(EXIT_FAILURE);
	}
	/* Multiply matrix with inverse */
	matrix_multiply(d_inv,d_mat,d_inv,n);
	
	int ur = is_unity_matrix(d_inv,n);
	if(ur)
		printf("SUCCESS! Matix inversion was successfull!!!!!\n");
	else
		printf("FAILED! Matrix inversion not successfull\n");
	
	free(h_mat);
	cudaFree(d_mat);
	cudaFree(d_inv);
}
