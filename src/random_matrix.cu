/* Matrix Inversion 
 * Group F: M. Lechner, P. Knöbel, J. Lövhall
 *
 * Library to generate a Random Matrix
*/

#include "includes.h"


/* Kernel that scales up and truncates the random variables */
static __global__
void ScaleUp(int n, float *mat, float max, int truncate)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < n && j < n) {
		float x = mat[i*n + j]*max;
		if(truncate)
			x = (float)((int)x);
		mat[i*n + j] = x;
	}
}

/* Allocates an array of size n-by-n on the device
 * and initializies it with random variables.
 * The random variables are in the range of (0, max]
 * If truncate != 0 the digits after the decimal point are truncated
 * i.e. instead of 5.38463 the variable will be 5.0000 
 */
float* random_matrix_generate(int n, float max, int truncate)
{
	int size = n*n;

	curandGenerator_t gen;
	float *d_mat;

	if(cudaMalloc((void **)&d_mat, size*sizeof(float)) != cudaSuccess)
	{
		return NULL;
	}

	/* Create pseudo-random number generator */
	if(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
	{
		cudaFree(d_mat);
		return NULL;
	}

	/* Set seed */
	if(curandSetPseudoRandomGeneratorSeed(gen, 273946962ULL) != CURAND_STATUS_SUCCESS)
	{
		curandDestroyGenerator(gen);
		cudaFree(d_mat);
		return NULL;
	}
	/* Generate n floats on device */
	if(curandGenerateUniform(gen, d_mat, size) != CURAND_STATUS_SUCCESS)
	{
		curandDestroyGenerator(gen);
		cudaFree(d_mat);
		return NULL;
	}

	/* Scale and truncate random variables */
	
	/* Let 16 by 16 threads run in parallel per block */
	dim3 threadsPerBlock(16, 16);
	
	int dimx = n / threadsPerBlock.x;
	int dimy = n / threadsPerBlock.y;
	
	/* Is n not divisible by 16 -> increment n by 1 to process the remaining elements */
	if( n > dimx * threadsPerBlock.x)
		dimx++;
	if( n > dimy * threadsPerBlock.y)
		dimy++;
	
	
	dim3 numBlocks(dimx, dimy);
	
	ScaleUp<<<numBlocks, threadsPerBlock>>>(n, d_mat, max, truncate);

	return d_mat;
}
