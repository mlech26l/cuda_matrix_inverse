#include "matrix_util.h"

#include<math.h>
#include<stdio.h>


#define TILE_WIDTH 16

/* This is the optimized matrix multiplication algorithm as discussed in the lecture.
   It uses the shared memory for caching
*/
// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
  		       int n) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < (n-1)/TILE_WIDTH+1; ++m) {
       if (Row < n && m*TILE_WIDTH+tx < n)
          ds_M[ty][tx] = A[Row*n + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < n && m*TILE_WIDTH+ty < n)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*n+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < n && Col < n)
       C[Row*n+Col] = Pvalue;
}

void mat_mul_dev( float* C, float* A, float* B, int n)
{
	
	 //@@ Initialize the grid and block dimensions here
    dim3 dimGrid((n-1)/TILE_WIDTH+1, (n-1)/TILE_WIDTH+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    //@@ Launch the GPU Kernel here
    matrixMultiply<<<dimGrid, dimBlock>>>(A, B, C,
                                          n);

}

__global__
void unity(int n, float *mat)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < n && j < n)
	{
		if(j==i)
		{
			mat[i*n + j] = 1;
		}
		// printf("%d, %d\n",i,j);
	}		
}
__global__
void setzero(int n, float *mat)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < n && j < n)
	{
		mat[i*n + j] = 0;
	}		

}

float* get_dev_unity_matrix(int n)
{
	int size = n*n;

	float *d_mat;

	if(cudaMalloc((void **)&d_mat, size*sizeof(float)) != cudaSuccess)
	{
		return NULL;
	}
	if(cudaMemset(d_mat, 0, size*sizeof(float)) != cudaSuccess)
	{
		return NULL;
	}
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
	
	// Setting all to 0 and the diagonal to 1 is done with 2 kernels, because of caching and memory access
	
	// Set all entries to 0 first, cannot use cudaMemset because of float (32 bit) data type
	setzero<<<numBlocks, threadsPerBlock>>>(n, d_mat);
	
	// Set the elements in the diagonal to 1
	unity<<<numBlocks, threadsPerBlock>>>(n, d_mat);

	return d_mat;

}

template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, int *g_odata, unsigned int n)
{
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	while(i + blockDim.x < n) { sdata[tid] += (int)g_idata[i] + (int)g_idata[i+blockSize]; i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int is_unity_matrix(float* d_mat, int n)
{
	int *h_sum = (int *)malloc(n*n* sizeof(int));
	
	int *d_sum;
	if(cudaMalloc((void**)&d_sum, n*n* sizeof(int)) != cudaSuccess)
	{
		printf("Error on Cuda Malloc!\n");
		return NULL;
	}
	if(cudaMemset(d_sum, 0,n*n* sizeof(int)) != cudaSuccess)
	{
		printf("Error on Cuda Memset!\n");
		return NULL;
	}

	/* Let 16 by 16 threads run in parallel per block */
	dim3 threadsPerBlock(16);
	
	int dimx = n*n / threadsPerBlock.x;
	
	/* Is n not divisible by 16 -> increment n by 1 to process the remaining elements */
	if( n > dimx * threadsPerBlock.x)
		dimx++;
	
	
	dim3 numBlocks(dimx);

	//reduce6<<<numBlocks, threadsPerBlock,64>>>(d_mat, d_sum,n,n*n);
	reduce6< 1><<< numBlocks, threadsPerBlock, 32 >>>(d_mat, d_sum,n*n); 
	
	if(cudaMemcpy(h_sum, d_sum, n*n* sizeof(int), cudaMemcpyDeviceToHost)!= cudaSuccess)
	{
		printf("Error at cudaMemcpy! ");
		printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
		exit(EXIT_FAILURE);
	}
	cudaFree(d_sum);
	int r = h_sum[0];
	free(h_sum);
	return r;
}