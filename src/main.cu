// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>


#include "device_query.h"
#include "testing_util.h"
#include "random_matrix.h"
#include "identity_matrix.h"
#include "matrix_multiplication.h"
#include "matrix.h"
#include "matrix_gpu.h"
#include "wingetopt.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
void jakobs_test_suite(int n);

void print_usage(char *progname)
{
  fprintf(stderr, "Usage: %s [-d deviceID] [-j] n\n",
          progname);
}
void run_matrix_inversion_test(int n, int use_gauss);
int main(int argc, char **argv)
{
  printf("CUDA Matrix inversion program - by Jakob and Mathias\n\n");
  int opt;
  int deviceID=0;
  int n=100;
  int use_gauss=0;

  while ((opt = getopt(argc, argv, "gn:d:")) != -1) {
          switch (opt) {
          case 'n':
              n = atoi(optarg);
              break;
          case 'd':
              deviceID = atoi(optarg);
              break;
          case 'g':
              use_gauss=1;
              break;
          default: /* '?' */
              print_usage(argv[0]);
              exit(EXIT_FAILURE);
          }
      }
  if(n<= 0)
  {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
  query_devices(deviceID);

  if(use_gauss)
  {
    jakobs_test_suite(n);
  }
  else
  {
    test_matrix_mathias_functions(n);
  }
	exit(EXIT_SUCCESS);
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

void WAprint(int size_of_one_side, float * matrix){
	printf("WA output form:\n");
	printf("inverse {");
	for(int x = 0; x < size_of_one_side; x++) {
		printf("{");
		for(int y = 0; y < size_of_one_side; y++) {
			printf("%1.0f", matrix[x*size_of_one_side + y]);
			if(y != size_of_one_side-1)
				printf(",");
		}
		printf("}");
		if(x != size_of_one_side-1)
			printf(",");
	}
	printf("}\n");
}

//simply check the bit patterns.. hope that the gpu uses the same precision as the cpu
int is_equal(float * a, float * b, int size){
	int i;
	for(i = 0;i < size;i++){
		if(abs(a[i] - b[i]) > 0.00001){
			return 0;
		}
	}
	return 1;
}

//do not touch this function if you do not really, really, know what you are doing.
void jakobs_test_suite(int n){
	printf("running Jakobs tests.\n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	float *matrix;
	float * matrix_org;
	printf("\nDoing matrix inversion test with n=%d\n",n);

	if(n == 3){
		matrix = create_identity_matrix(n);//(float *)malloc(n*n* sizeof(float));
		matrix[1] = 1;
		matrix[6] = 1;
		matrix_org = create_identity_matrix(n); //used instead of malloc because lazy and easy..
	} else {
		matrix = (float *)malloc(sizeof(float)*n*n);
		matrix_org = (float *)malloc(sizeof(float)*n*n);
		float * d_mat;
		d_mat = generate_random_matrix(n,100,1);
		gpuErrchk(cudaMemcpy(matrix, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToHost))
	}


	int i;
	for(i = 0;i <n*n; i++){
		matrix_org[i] = matrix[i];
	}
	float* inverse = create_identity_matrix(n);
	float* inverse_matrix_cpu = create_identity_matrix(n);

	/* Print out test matrix */
	if(n == 3){
		printf("test Matrix org:\n");
		print_matrix(matrix,n);

		printf("test Matrix:\n");
		print_matrix(matrix,n);
		WAprint(n,matrix);
	}

	int succ=0;
	{
		cudaEventRecord(start);
    cudaEventSynchronize(start);


		//running cpu test first because it has singularity check.
		//inversion destroys the matrix
		inverse_cpu(matrix, n, inverse_matrix_cpu, &succ);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
		printf("CPU inverse took seconds: %f\n", time);
	}

	if(!succ)
	{
		printf("Matrix singular!");
		exit(EXIT_SUCCESS);
	}

	//restore the matrix
	for(i = 0;i <n*n; i++){
		matrix[i] = matrix_org[i];
	}

	{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventSynchronize(start);
		inverse_gpu(matrix, n, inverse, &succ);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

		printf("CUDA inverse took seconds: %f\n", time);
	}

	if(!is_equal(inverse,inverse_matrix_cpu,n*n)){
		printf("matrixes not equal. printing.\n\n");
		printf("gpu matrix \n");

		print_matrix(inverse, n);
		printf("\n\ncpu matrix \n");
		print_matrix(inverse_matrix_cpu, n);

		printf("start matrix\n");
		WAprint(n,matrix_org);
	} else {
		printf("matrixes equal. all is good. \n");
	}


	free(matrix);
	free(matrix_org);
	free(inverse);
	free(inverse_matrix_cpu);
}
