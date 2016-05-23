#include "unity_matrix.h"

#include<math.h>
#include<stdio.h>


#define TILE_WIDTH 16

// /* This is the optimized matrix multiplication algorithm as discussed in the lecture.
   // It uses the shared memory for caching
// */
// Compute C = A * B
// __global__ void matrixMultiply(float * A, float * B, float * C,
  		       // int n) {
    // @@ Insert code to implement matrix multiplication here
    // __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    // __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    // int bx = blockIdx.x, by = blockIdx.y,
       // tx = threadIdx.x, ty = threadIdx.y,
       // Row = by * TILE_WIDTH + ty,
       // Col = bx * TILE_WIDTH + tx;
    // float Pvalue = 0;

    // for (int m = 0; m < (n-1)/TILE_WIDTH+1; ++m) {
       // if (Row < n && m*TILE_WIDTH+tx < n)
          // ds_M[ty][tx] = A[Row*n + m*TILE_WIDTH+tx];
       // else
          // ds_M[ty][tx] = 0;
       // if (Col < n && m*TILE_WIDTH+ty < n)
          // ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*n+Col];
       // else
          // ds_N[ty][tx] = 0;

       // __syncthreads();
       // for (int k = 0; k < TILE_WIDTH; ++k)
          // Pvalue += ds_M[ty][k] * ds_N[k][tx];
       // __syncthreads();
    // }
    // if (Row < n && Col < n)
       // C[Row*n+Col] = Pvalue;
// }

// void mat_mul_dev( float* C, float* A, float* B, int n)
// {

	 // @@ Initialize the grid and block dimensions here
    // dim3 dimGrid((n-1)/TILE_WIDTH+1, (n-1)/TILE_WIDTH+1, 1);
    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // @@ Launch the GPU Kernel here
    // matrixMultiply<<<dimGrid, dimBlock>>>(A, B, C,
                                          // n);

// }

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

__device__
void accumulate(float mat, int i, int n, int &ret)
{
	int x=i/n;
	int y = i%n;
	if(y==x)
	{
		if(mat>1.001 || mat<0.99)
			ret = 1;
		else ret = 0;
	}
	else
	{
		if(mat>0.01 || mat < -0.01)
			ret = 1;
		else ret = 0;
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

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};
template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(float *g_idata, int *g_odata, int size, int n)
{
    int *sdata = SharedMemory<int>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    int mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < size)
    {
		int acc=0;
		accumulate(g_idata[i], i, n, acc);
        mySum += acc;

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < size)
		{
			int acc2=0;
			accumulate(g_idata[i+blockSize], i+blockSize, n, acc2);
			mySum += acc2;
            mySum += acc2;
		}

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}


#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = MIN(maxBlocks, blocks);
    }
}
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

void
reduce(int n, int threads, int blocks, float *d_idata, int *d_odata)
{
	int size = n*n;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);


	if (isPow2(size))
	{
		switch (threads)
		{
			case 512:
				reduce6<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 256:
				reduce6<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 128:
				reduce6<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 64:
				reduce6<64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 32:
				reduce6< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 16:
				reduce6< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case  8:
				reduce6<   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case  4:
				reduce6<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case  2:
				reduce6<   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case  1:
				reduce6<   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;
		}
	}
	else
	{
		switch (threads)
		{
			case 512:
				reduce6< 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 256:
				reduce6< 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 128:
				reduce6< 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 64:
				reduce6<  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 32:
				reduce6<  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case 16:
				reduce6<  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case  8:
				reduce6<   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case  4:
				reduce6<   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case  2:
				reduce6<   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;

			case  1:
				reduce6<   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
				break;
		}
	}

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

	int numBlocks=8;
	int numThreads = 32;
	reduce(n, numThreads, numBlocks, d_mat, d_sum);

	if(cudaMemcpy(h_sum, d_sum, numBlocks* sizeof(int), cudaMemcpyDeviceToHost)!= cudaSuccess)
	{
		printf("Error at cudaMemcpy! ");
		printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
		exit(EXIT_FAILURE);
	}
	printf("sum matrix:\n");
	for(int x = 0; x < numBlocks; x++) {
			printf("%d ", h_sum[x]);	
	}
	printf("\n");

	int ret =0;
	for(int i=0;i<numBlocks;i++)
	{
		ret+= h_sum[i];
	}
	printf("numBlocks: %d\n",numBlocks);


	cudaFree(d_sum);
	free(h_sum);
	return ret;
}