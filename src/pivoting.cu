#include<math.h>
#include<stdio.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "pivoting.h"

static pivoting_max_entry* reduced_block;
static pivoting_max_entry *host_block;
static int reduced_block_size;
static int maxThreadsPerBlock;
static int maxGridSize;

#define cudaCheck(ans) do{if(ans != cudaSuccess){fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(ans),  __FILE__, __LINE__); exit(EXIT_FAILURE);} }while(false)

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

__device__ void check_max(float value, int index, pivoting_max_entry &obj)
{
  // printf(" Thx(%d) Casting to struct: %1.3f, %d\n",threadIdx.x ,value,index);
obj.value=value;
obj.index=index;
}
__device__  void combine_max(pivoting_max_entry &a, pivoting_max_entry &b)
{
  //  printf("Thx(%d) Combining %d, %d (%1.3f, %1.3f)\n",threadIdx.x, a.index,b.index, a.value, b.value);
  if(fabs(b.value)>fabs(a.value))
  {
    a.value=b.value;
    a.index = b.index;
  }
}
template <unsigned int blockSize>
__global__ void
reduce_max(float *g_idata, pivoting_max_entry *g_odata, int size, int n)
{
    pivoting_max_entry *sdata = SharedMemory<pivoting_max_entry>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    pivoting_max_entry local_max;
    local_max.index=0;
    local_max.value=0.0f;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < size)
    {
        // printf("access item at %d (%1.3f)\n",i*n,g_idata[i*n]);
		    pivoting_max_entry acc;
        check_max(g_idata[i*n], i, acc);
        combine_max(local_max, acc);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < size)
		{
			pivoting_max_entry acc2;

        // printf("access item at %d (%1.3f)\n",(i+blockSize)*n,g_idata[(i+blockSize)*n]);
			check_max(g_idata[(i+blockSize)*n], (i+blockSize), acc2);
      combine_max(local_max,acc2);
		}

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = local_max;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        combine_max(local_max, sdata[tid + 256]);
        sdata[tid] = local_max;
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
             combine_max( local_max, sdata[tid + 128]);
             sdata[tid] = local_max;
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       combine_max( local_max , sdata[tid +  64]);
       sdata[tid] = local_max;
    }

    __syncthreads();


    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
       combine_max(local_max , sdata[tid + 32]);
       sdata[tid] = local_max;
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
         combine_max(local_max , sdata[tid + 16]);
         sdata[tid] = local_max ;
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
         combine_max(local_max , sdata[tid +  8]);
         sdata[tid] = local_max;
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        combine_max( local_max , sdata[tid +  4]);
        sdata[tid] = local_max;
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
         combine_max(local_max , sdata[tid +  2]);
         sdata[tid] = local_max;
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
         combine_max(local_max , sdata[tid +  1]);
         sdata[tid] = local_max;
    }

    __syncthreads();


    // write result for this block to global mem

    if (tid == 0){
        //  printf("Write result %1.3f to block (%d)\n",local_max.value,blockIdx.x);
       g_odata[blockIdx.x] = local_max;
     }
}


#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

static unsigned int nextPow2(unsigned int x)
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
static void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);


    if ((float)threads*blocks > (float)maxGridSize * maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > maxGridSize)
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, maxGridSize, threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = MIN(maxBlocks, blocks);
    }
}

void reduce_max_host(int n, int threads, int blocks, float *d_idata, int size, pivoting_max_entry *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(pivoting_max_entry) : threads * sizeof(pivoting_max_entry);

	switch (threads)
	{
		case 512:
			reduce_max< 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;

		case 256:
			reduce_max< 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;

		case 128:
			reduce_max< 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;

		case 64:
			reduce_max<  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;

		case 32:
			reduce_max<  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;

		case 16:
			reduce_max<  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;

		case  8:
			reduce_max<   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;

		case  4:
			reduce_max<   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;

		case  2:
			reduce_max<   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;

		case  1:
			reduce_max<   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,n);
			break;
	}
}

pivoting_max_entry pivoting_find_pivot_semi_gpu(float *A, int n, int row)
{
  int blocks, threads;

  int size=n-row;
  // DO not use threads > 32!!!!!! -> strange behaviour -> cudaMemcpy will fail
  getNumBlocksAndThreads(6, size, 1000, 32, blocks, threads);

  //void reduce_max_host(int n, int threads, int blocks, float *d_idata, int row, pivoting_max_entry *d_odata)

  // printf("Launch redcution kernel <<%d, %d>>\n",blocks,threads);
  reduce_max_host(n,threads,blocks,A+row*(1+n),size,reduced_block);
  // Allocate block size of memory on host


	// Copy last block to host
	cudaCheck(cudaMemcpy(host_block, reduced_block, blocks* sizeof(pivoting_max_entry), cudaMemcpyDeviceToHost));

	// Process last block
	pivoting_max_entry ret;
  ret=host_block[0];
	for(int i=1;i<blocks;i++)
	{
    // printf("Block res: %1.3f, %d\n",ret.value,ret.index);
    if(fabs(host_block[i].value) > fabs(ret.value))
		  ret = host_block[i];
	}

  ret.index+=row;
  return ret;
}

void pivoting_preload_device_properties(int n)
{
  reduced_block_size = n/32;
  if(n > reduced_block_size*32)
    reduced_block_size++;

  cudaCheck(cudaMalloc((void**)&reduced_block, reduced_block_size* sizeof(pivoting_max_entry)));
  host_block = (pivoting_max_entry *)malloc(reduced_block_size* sizeof(pivoting_max_entry));
  //get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  maxThreadsPerBlock = prop.maxThreadsPerBlock;
  maxGridSize = prop.maxGridSize[0];
}
void pivoting_unload_device_properties(void)
{
  cudaCheck(cudaFree(reduced_block));
  free(host_block);
}
