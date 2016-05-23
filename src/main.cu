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

int main(int argc, char **argv)
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
		return NULL;
	}
	if(cudaMemcpy(d_b, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToDevice)!= cudaSuccess)
	{
		printf("Error at cudaMalloc! ");
		exit(EXIT_FAILURE);
	}
	
	if(cudaMalloc((void **)&d_c, n*n*sizeof(float)) != cudaSuccess)
	{
		printf("Error on Cuda Malloc!\n");
		return NULL;
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
	exit(EXIT_SUCCESS);

}
