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

int main(int argc, char **argv)
{
	
	float *h_mat, *d_mat;
	int n = 10;
	
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
	
	free(h_mat);
	cudaFree(d_mat);
	exit(EXIT_SUCCESS);

}
