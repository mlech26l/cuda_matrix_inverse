#include "matrix_multiplication.h"
#include <stdio.h>

// Divides, if not evenly then increment quotient
static int divup(int z, int n)
{
	int d = z/n;
	if(n*d != z)
	d++;
	return d;
}
/**
* Matrix multiplication: C = A * B.
* Host code.
*
* This sample implements matrix multiplication as described in Chapter 3
* of the programming guide.
* It has been written for clarity of exposition to illustrate various CUDA
* programming principles, not with the goal of providing the most
* performant generic kernel for matrix multiplication.
*
* See also:
* V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
* in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
* Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
*/

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
* A,B and C are n-by-n matrices
*/

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int n)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = n * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + n - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;


	// Step size used to iterate through the sub-matrices of B
	int bStep  = BLOCK_SIZE * n;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	int py = BLOCK_SIZE * by + ty;
	int px = BLOCK_SIZE * bx + tx;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
	{
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		int ixa = a + n * ty + tx;
		int ixb = b + n * ty + tx;

		int ixa_x = ixa/n;
		int ixa_y = ixa%n;

		int ixb_x = ixb/n;
		int ixb_y = ixb%n;
		if(ixa_x<n && ixa_y<n)
		{
			As[ty][tx] = A[ixa];
		}
		else{
			As[ty][tx] = 0;
		}
		if(ixb_x<n && ixb_y<n)
		{
			Bs[ty][tx] = B[ixb];
		}
		else{
			Bs[ty][tx] = 0;
		}

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
		//	printf("C[%d,%d] += %1.4f\n",px,py,As[ty][k] * Bs[k][tx]);
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c =  py * n + px;
	if(px < n && py < n)
	{
		C[c] = Csub;
		// printf("C[%d,%d]=...\n",px,py);
	}
}


void matrix_multiplication(float* C, float* A, float* B, int n)
{
	// Define grid
	dim3 dimBlock(MULTIPLY_BLOCK_SIZE, MULTIPLY_BLOCK_SIZE);
	dim3 dimGrid(divup(n,dimBlock.x), divup(n,dimBlock.y));


	// printf("Launching Multiplication grid <<%d, %d>, <%d, %d>>\n",dimGrid.x,dimGrid.y,dimBlock.x,dimBlock.y);

	// Launch kernel
	matrixMulCUDA<MULTIPLY_BLOCK_SIZE><<<dimGrid, dimBlock>>>(C, A, B, n);


}
