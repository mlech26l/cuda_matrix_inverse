// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

#include "testing_util.h"
#include "random_matrix.h"
#include "identity_matrix.h"
#include "matrix_multiplication.h"
#include "matrix.h"
#include "matrix_gpu.h"
#include "gpu_enabled.h"
#include "gpu_pivoting.h"
#include "device_query.h"

#define cudaCheck(ans) do{if(ans != cudaSuccess){fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(ans),  __FILE__, __LINE__); exit(EXIT_FAILURE);} }while(false)



__global__ void print_matrix_on_device_kernel(float* d_A, int n)
{
  for(int x=0;x<n;x++)
  {
    for(int y =0;y<n;y++)
    {
      printf("%1.3f, ",d_A[x*n+y]);
    }
    printf("\n");
  }
}


// Function for testing the matrix multiplication and check for identity matrix
void test_gpu_pivoting(void);


static void do_complete_check(float *d_mat,float* d_mat2, float *d_inv, float* h_inv, int n);
static int check_first_elements_for_identity(float *A, float *Ainv, int n)
{
  int sub=5;
	for(int x = 0; x < sub; x++) {
		for(int y = 0; y < sub; y++) {
			float sum=0;
			for(int k=0;k<n;k++)
			{
				sum+= A[x*n+k]*Ainv[k*n+y];
			}
      if(x==y)
      {
        if( sum<0.99||sum>1.01 )
        {
          return 0;
        }
      }
			else{
        if(sum<-0.01 || sum >0.01)
        {
          return 0;
        }
      }
		}
	}
  return 1;

}
static void do_partial_check(float *d_inv, float* h_inv, float *h_mat,float *d_mat2, int n)
{
  printf("Doing partial check for identity!\n");
  cudaCheck(cudaMemcpy(h_mat,d_mat2,sizeof(float)*n*n,cudaMemcpyDeviceToHost));

  printf("Checking CPU matrix for identity ... ");
  if(check_first_elements_for_identity(h_mat,h_inv,n))
  {
    printf("[SUCCESS]\n");
  }
  else printf("[FAIL]\n");

  cudaCheck(cudaMemcpy(h_inv,d_inv,sizeof(float)*n*n,cudaMemcpyDeviceToHost));
  printf("Checking GPU matrix for identity ... ");
  if(check_first_elements_for_identity(h_mat,h_inv,n))
  {
    printf("[SUCCESS]\n");
  }
  else printf("[FAIL]\n");
}
void test_gpu_pivoting(void){
  printf("**** Start GPU pivoting! ****\n");

  int n=5;
  float *d_vec;
  float* h_vec;
  d_vec =  generate_random_matrix(n,100,1);
  h_vec=(float*)malloc(sizeof(float)*n*n);
  if(d_vec == NULL)
  {
    printf("Error could not allocate random matrix of size (%d)\n",n);
    exit(EXIT_FAILURE);
  }
  float max_value_c = 200.0f;
  cudaCheck(cudaMemcpy(d_vec+3, &max_value_c, sizeof(float), cudaMemcpyHostToDevice));

  cudaCheck(cudaMemcpy(h_vec,d_vec,sizeof(float)*n*n,cudaMemcpyDeviceToHost));
  printf("Matrix\n");
  for(int x=0;x<n;x++)
  {
    for(int y=0;y<n;y++)
    {
      printf("%1.3f, ",h_vec[x*n+y]);
    }
    printf("\n");
  }
  max_entry ret = find_pivot_semi_gpu(d_vec,n,3);
    printf("Max element: %1.4f, at (%d)\n",ret.value,ret.index);
  cudaCheck(cudaFree(d_vec));
  printf("**** GPU pivoting test finished! ****\n\n");
  exit(EXIT_SUCCESS);
}
void test_matrix_mathias_functions(int n)
{
  /* 100MB Matrix (5000x5000) take 4min to invert on GPU (GT 525m, with 96 threads)
  vs 28 min on a Intel i5 2430m core (= 7 x speedup)
  */
  printf("Mathias test utility!\n");
  //test_gpu_pivoting();
	float *d_mat, *d_inv, *d_mat2;
	float *h_mat, *h_inv;

	printf("\nDoing LU matrix inversion test with n=%d\n",n);


	d_mat = generate_random_matrix(n,100,1);
  cudaCheck(cudaMalloc((void**)&d_inv, sizeof(float)*n*n));
  cudaCheck(cudaMalloc((void**)&d_mat2, sizeof(float)*n*n));

  h_mat=(float*)malloc(sizeof(float)*n*n);
  h_inv=(float*)malloc(sizeof(float)*n*n);
  cudaCheck(cudaMemcpy(h_mat, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToHost));

	/* Copy random matrix on device */
	cudaCheck(cudaMemcpy(d_mat2, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToDevice));


  if(lup_matrix_inverse_gpu(d_mat, d_inv, n)==0)
	{
		printf("Matrix singular!");
		exit(EXIT_SUCCESS);
	}

  lup_matrix_inverse_cpu(h_mat,h_inv,n);

  // Do check if inversion was successfull
  if(n<=900)
  {
    do_complete_check(d_mat,d_mat2, d_inv, h_inv, n);
  }
  else{
    do_partial_check(d_inv, h_inv, h_mat, d_mat2,n);
  }


	cudaCheck(cudaFree(d_mat));
	cudaCheck(cudaFree(d_mat2));
	cudaCheck(cudaFree(d_inv));
  free(h_inv);
  free(h_mat);
  cudaProfilerStop();
}

static void do_complete_check(float *d_mat,float* d_mat2, float *d_inv, float* h_inv, int n)
{
  printf("Doing complete check for Identity:\n");
    cudaEvent_t start, stop;
    float milliseconds;
  	float *d_identity;
  	cudaCheck(cudaMalloc((void **)&d_identity, n*n* sizeof(float)));

  	/* Multiply matrix with inverse */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    printf("Matrix multipying...");
  	matrix_multiply(d_identity,d_mat2,d_inv,n);
    printf("[OK]");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf(" (in %1.3f ms)\n",milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // printf("Identity matrix:\n");
    // print_matrix_on_device_kernel<<<1,1>>>(d_identity,n);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    printf("Check for Identity matrix...");
  	int ur = is_identity_matrix(d_identity,n);
    printf("[OK]");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf(" (in %1.3f ms)\n",milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  	if(ur)
  		printf("SUCCESS! Matix inversion was successfull!!!!!\n");
  	else
  		printf("FAILED! Matrix inversion not successfull\n");


    cudaCheck(cudaMemcpy(d_inv, h_inv, n*n * sizeof(float), cudaMemcpyHostToDevice));
    printf("Matrix from host multipying...");
  	matrix_multiply(d_identity,d_mat2,d_inv,n);
    printf("[OK]\n");

    // printf("Identity matrix:\n");
    // print_matrix_on_device_kernel<<<1,1>>>(d_identity,n);

    printf("Check host matrix for Identity...");
  	ur = is_identity_matrix(d_identity,n);
    printf("[OK]\n");
  	if(!ur)
  		printf("Host matrix inversion failed!\n");
}
