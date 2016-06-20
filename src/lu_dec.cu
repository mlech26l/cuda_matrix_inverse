/* Matrix Inversion 
 * Group F: M. Lechner, P. Knöbel, J. Lövhall
 *
 * LU decomposition implementation of Matrix Inversion  
*/

#include "includes.h"


static int find_pivot_cpu(float *A, int n, int row) {
  int pivot = row;
  float max = fabs(A[row * n + row]);
  for (int j = row + 1; j < n; j++) {
    if (max < fabs(A[j * n + row])) {
      max = fabs(A[j * n + row]);
      pivot = j;
    }
  }
  return pivot;
}
static void swap_item_kernel_cpu(float *A, int n, int a, int b, int thx) {
  // int thx = ...
  if (thx < n) {
    float temp = A[a * n + thx];
    A[a * n + thx] = A[b * n + thx];
    A[b * n + thx] = temp;
  }
}
static void swap_rows_cpu(float *A, int n, int a, int b) {
  // replace with grid
  for (int j = 0; j < n; j++) {
    swap_item_kernel_cpu(A, n, a, b, j);
  }
}
static void divide_row_kernel_cpu(float *A, int n, int k, float denominator,
                              int thx) {
  // int thx = ...
  int j = k+1+thx;
  if (j < n) {
    A[k * n + j] /= denominator;
  }
}
static void divide_row_cpu(float *A, int n, int k, float denominator) {
  for (int j = 0; j < n-k - 1; j++) {
    divide_row_kernel_cpu(A, n, k, denominator, j);
  }
}
static void update_sub_matrix_kernel_cpu(float *A, int n, int k, int thx, int thy) {

  int i = thx + k + 1;
  int j = thy + k + 1;
  if (i < n && j< n) {
    A[i * n + j] -= A[i * n + k] * A[k * n + j];
  }
}
static void update_sub_matrix_cpu(float *A, int n, int k) {
  for (int i = 0; i < n-k-1; i++) {
    for (int j = 0; j < n-k-1; j++) {
      update_sub_matrix_kernel_cpu(A, n, k, i, j);
    }
  }
}
static int lup_decomposition_cpu(float *A, int *pivot, int n) {
  for (int k = 0; k < n; k++) {
    pivot[k] = find_pivot_cpu(A, n, k);

    // printf("Step %d CPU LU matrix:\n",k);
    // for(int x=0;x<n;x++)
    // {
    //   for(int y =0;y<n;y++)
    //   {
    //     printf("%1.3f, ",A[x*n+y]);
    //   }
    //   printf("\n");
    // }

    float denominator = A[pivot[k] * n + k];
    if (denominator > -0.0001 && denominator < 0.0001) {
      // determinant close to 0 -> matrix singular
      return -1;
    }
    // printf("*** CPU Step %d, pivot: %d, denom: %1.3f\n",k,pivot[k],denominator);

    if (pivot[k] != k)
      swap_rows_cpu(A, n, k, pivot[k]);

    divide_row_cpu(A, n, k, denominator);

    update_sub_matrix_cpu(A, n, k);
  }
  return 0;
}

static void lup_invert_column_cpu(float *LU, int col, int *pivot, float *x, int n) {
  int b = col;

  // Solves LU *x = pivot(b), where b is col-th column of the identity matrix
  // Part 1: Solving L*y = pivot(b)
  for (int k = 0; k < n; k++) {

    if (pivot[k] != k) {
      if (k == b) {
        b = pivot[k];
      } else if (pivot[k] == b) {
        b = k;
      }
    }

    //https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
    // x(k) = b(k)
    x[k * n] = 0.0f;
    if (k == b)
      x[k * n] = 1.0f;


    for (int i = 0; i < k; i++)
      x[k * n] -= x[i * n] * LU[n * k + i];
    x[k * n] /= LU[n * k + k];
  }

  // Part 2: Solving U* x = y
  // (n-1)-th element (=last) is already result
  // Need to subtract result from rows (n-2) to 0
  for (int k = n - 1; k >= 0; k--) {
    for (int i = k + 1; i < n; i++)
      x[k * n] -= x[i * n] * LU[k * n + i];
  }
}

int lu_dec_matrix_inverse_cpu(float *A, float* A1,int n)
{
  cudaEvent_t start, stop;
  float milliseconds=0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cudaEventSynchronize(start);
  printf("Decompose LU on host...");
  int *P = (int *)malloc(sizeof(int) * n);
  if (lup_decomposition_cpu(A, P, n) < 0) {
    printf("Matrix singular!\n");
    free(P);
    return 0;
  }
  cudaEventRecord(stop);
  printf("[OK]");
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf(" (in %1.3f ms)\n",milliseconds);


  // printf("LU CPU matrix: \n");
  // for(int x=0;x<n;x++)
  // {
  //   for(int y =0;y<n;y++)
  //   {
  //     printf("%1.3f, ",A[x*n+y]);
  //   }
  //   printf("\n");
  // }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaEventSynchronize(start);
  printf("Inverting matrix on host...");
  for (int i = 0; i < n; i++) {
    lup_invert_column_cpu(A, i, P, &A1[i], n);
  }
  cudaEventRecord(stop);
  printf("[OK]");
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf(" (in %1.3f ms)\n",milliseconds);


  //
  // printf("Inverse CPU matrix: \n");
  // for(int x=0;x<n;x++)
  // {
  //   for(int y =0;y<n;y++)
  //   {
  //     printf("%1.3f, ",A1[x*n+y]);
  //   }
  //   printf("\n");
  // }

  free(P);
  return 1;
}



__global__ void print_kernel(float* A, int n)
{
  for(int x=0;x<n;x++)
  {
    for(int y =0;y<n;y++)
    {
      printf("%1.3f, ",A[x*n+y]);
    }
    printf("\n");
  }
}

__global__ void lup_invert_column_gpu_kernel(float *LU, int *pivot, float *A1, int n);
void lup_invert_column_gpu(float *LU, int *P, float *A1, int n) {

  // for(int k=0;k<n;k++)
  // {
  //
  // }
  int threads=32;
  int blocks = n/threads;
  if(n > threads*blocks)
    blocks++;

  lup_invert_column_gpu_kernel<<<blocks,threads>>>(LU,P, A1,n);
}
// for (int i = 0; i < n; i++) {
//   lup_invert_column_gpu(A, i, P2, &A1[i], n);
// }
__global__ void lup_invert_column_gpu_kernel(float *LU, int *pivot, float *A1, int n) {

  int col = threadIdx.x+blockIdx.x*32;

  if(col<n)
  {
    float *x = A1+col;
    int b = col;

    for (int k = 0; k < n; k++) {

      if (pivot[k] != k) {
        if (k == b) {
          b = pivot[k];
        } else if (pivot[k] == b) {
          b = k;
        }
      }

      x[k * n] = 0.0f;
      if (k == b)
        x[k * n] = 1.0f;

      for (int i = 0; i < k; i++)
        x[k * n] -= x[i * n] * LU[n * k + i];
      x[k * n] /= LU[n * k + k];
    }

    for (int k = n - 1; k >= 0; k--) {
      for (int i = k + 1; i < n; i++)
        x[k * n] -= x[i * n] * LU[k * n + i];
    }
  } // end of kernel
}



__global__ static void swap_item_kernel(float *A, int n, int a, int b) {
  // int thx = ...
  int thx = threadIdx.x + blockIdx.x*32;
  if (thx < n) {
    float temp = A[a * n + thx];
    A[a * n + thx] = A[b * n + thx];
    A[b * n + thx] = temp;
  }
}
static void swap_rows(float *A, int n, int a, int b) {
  int threads=32;

  int blocks = n / threads;

  /* Is n not divisible by 32 -> increment n by 1 to process the remaining elements */
  if( n > threads * blocks)
    blocks++;

    swap_item_kernel<<<blocks,threads>>>(A, n, a, b);
}
__global__ static void divide_row_kernel(float *A, int n, int k, float denominator) {
  // int thx = ...
  int thx = threadIdx.x + blockIdx.x*32;
  int j = k+1+thx;
  if (j < n) {
    A[k * n + j] /= denominator;
  }
}
static void divide_row(float *A, int n, int k, float denominator) {
  int size = n-k - 1;
  if(size == 0 )
    return;

  int threads=32;

  int blocks = size / threads;

  /* Is n not divisible by 32 -> increment n by 1 to process the remaining elements */
  if( size > threads * blocks)
    blocks++;

    divide_row_kernel<<<blocks, threads>>>(A, n, k, denominator);
}
__global__ static void update_sub_matrix_kernel(float *A, int n, int k) {

  int thx = threadIdx.x + blockIdx.x*16;
  int thy = threadIdx.y + blockIdx.y*16;

  int i = thx + k + 1;
  int j = thy + k + 1;
  if (i < n && j < n) {
    A[i * n + j] -= A[i * n + k] * A[k * n + j];
  }
}
static void update_sub_matrix(float *A, int n, int k) {
  int size = n-k-1;

  if(size==0)
    return;
  /* Let 16 by 16 threads run in parallel per block */
	dim3 threadsPerBlock(16, 16);

	int dimx = size / threadsPerBlock.x;
	int dimy = size / threadsPerBlock.y;

	/* Is n not divisible by 16 -> increment n by 1 to process the remaining elements */
	if( size > dimx * threadsPerBlock.x)
		dimx++;
	if( size > dimy * threadsPerBlock.y)
		dimy++;


	dim3 numBlocks(dimx, dimy);
  update_sub_matrix_kernel<<<numBlocks, threadsPerBlock>>>(A, n, k);
}

static int lup_decomposition_gpu(float *A, int *pivot, int n) {

  for (int k = 0; k < n; k++) {

    pivoting_max_entry pivoting = pivoting_find_pivot_semi_gpu(A,n,k);
    pivot[k] = pivoting.index;


    // printf("Step %d LU GPU matrix:\n",k);
    // print_kernel<<<1,1>>>(A,n);

    float denominator = pivoting.value;
    if (denominator > -0.0001 && denominator < 0.0001) {
      // determinant close to 0 -> matrix singular
      return -1;
    }

    // printf("*** GPU Step %d, pivot: %d, denom: %1.3f\n",k,pivot[k],denominator);
    if (pivot[k] != k)
      swap_rows(A, n, k, pivot[k]);

    divide_row(A, n, k, denominator);

    update_sub_matrix(A, n, k);
  }
  return 0;
}




int lu_dec_matrix_inverse_gpu(float *d_A, float* d_A1,int n)
{
  pivoting_preload_device_properties(n);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  int *h_P = (int *)malloc(sizeof(int) * n);
  printf("Decomposing into LU...");
  cudaEventRecord(start);
  cudaEventSynchronize(start);
  if (lup_decomposition_gpu(d_A, h_P, n) < 0) {
    free(h_P);
    return 0;
  }
  cudaEventRecord(stop);
  printf("[OK]");
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf(" (in %1.3f ms)\n",milliseconds);
  // printf("LU GPU matrix:\n");
  // print_kernel<<<1,1>>>(d_A,n);

  int  *d_P;
  cudaCheck(cudaMalloc((void**)&d_P, n* sizeof(int)));

  cudaCheck(cudaMemcpy(d_P,h_P,sizeof(int)*n, cudaMemcpyHostToDevice));


  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  printf("Inverting LU matrix...");
  cudaEventRecord(start);
  cudaEventSynchronize(start);
  lup_invert_column_gpu(d_A,d_P,d_A1,n);
  cudaEventRecord(stop);
  printf("[OK]");
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf(" (in %1.3f ms)\n",milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // printf("Inverse GPU matrix:\n");
  // print_kernel<<<1,1>>>(d_A1,n);



  free(h_P);
  cudaCheck(cudaFree(d_P));
  pivoting_unload_device_properties();

  return 1;
}





