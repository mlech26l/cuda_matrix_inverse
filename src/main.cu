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
#include "matrix.h"
#include "matrix_gpu.h"
#include "lup_decomposition.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
void jakobs_test_suite();
static void multiply_and_print(float* A, float* Ainv, int n);

// Function for testing the matrix multiplication and check for unity matrix
void test_matrix_util_functions(void);

int main(int argc, char **argv)
{
	if(argc > 1){
		if(!strcmp(argv[1],"-j")){ //this is so that nobody removes the gpu test again.. don't touch this.
			jakobs_test_suite();
			return 0;
		}
	}
	test_matrix_util_functions();
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
		if(a[i] != b[i]){
			return 0;
		}
	}
	return 1;
}

//do not touch this function if you do not really, really, know what you are doing.
void jakobs_test_suite(){
	printf("running Jakobs tests.\n");
	float *matrix;
	int n = 3;
	printf("\nDoing matrix inversion test with n=%d\n",n);

	/* Allocate n floats on host */
	matrix = create_identity_matrix(n);//(float *)malloc(n*n* sizeof(float));
	matrix[1] = 1;
	matrix[6] = 1;
	float * matrix_org = create_identity_matrix(n); //used instead of malloc because lazy and easy..
	int i;
	for(i = 0;i <n*n; i++){
		matrix_org[i] = matrix[i];
	}
	float* inverse = create_identity_matrix(n);
	float* inverse_matrix_cpu = create_identity_matrix(n);
	/* Allocate n floats on device */

	/* Print out test matrix */
	printf("test Matrix org:\n");
	print_matrix(matrix,n);

	printf("test Matrix:\n");
	print_matrix(matrix,n);
	WAprint(n,matrix);

	int succ=0;

	//running cpu test first because it has singularity check.
	//inversion destroys the matrix
	inverse_cpu(matrix, n, inverse_matrix_cpu, &succ);
	if(!succ)
	{
		printf("Matrix singular!");
		exit(EXIT_SUCCESS);
	}

	//restore the matrix
	for(i = 0;i <n*n; i++){
		matrix[i] = matrix_org[i];
	}

	inverse_gpu(matrix, n, inverse, &succ);

	if(!is_equal(inverse,inverse_matrix_cpu,n*n)){
		printf("matrixes not equal. printing.\n\n");
		printf("gpu matrix \n");
		print_matrix(inverse, n);
		printf("\n\ncpu matrix \n");
		print_matrix(inverse_matrix_cpu, n);
	} else {
		printf("matrixes equal. all is good. \n");
	}


	free(matrix);
	free(matrix_org);
	free(inverse);
	free(inverse_matrix_cpu);
}

void test_matrix_util_functions(void)
{
	float *h_mat, *d_mat;
	int n = 6;
	printf("Enter matrix dim: ");
	scanf("%d",&n);
	printf("\nDoing matrix inversion test with n=%d\n",n);

	/* Allocate n floats on host */
	h_mat = (float *)malloc(n*n* sizeof(float));
	float* h_inv = create_identity_matrix(n);
	/* Allocate n floats on device */

	d_mat = generate_random_matrix(n,100,1);

	/* Copy random matrix to host */
	gpuErrchk(cudaMemcpy(h_mat, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToHost))

	/* Print out random generated matrix */
	printf("Random generated Matrix:\n");
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_mat[x*n + y]);
		}
		printf("\n");
	}
	printf("WA output form:\n");
	printf("inverse {");
	for(int x = 0; x < n; x++) {
		printf("{");
		for(int y = 0; y < n; y++) {
			printf("%1.0f", h_mat[x*n + y]);
			if(y != n-1)
				printf(",");
		}
		printf("}");
		if(x != n-1)
			printf(",");
	}
	printf("}\n");

	int succ=0;
	inverse_cpu(h_mat, n, h_inv, &succ);
	if(!succ)
	{
		printf("Matrix singular!");
		exit(EXIT_SUCCESS);
	}
	/* Copy random matrix to host for checking */
	gpuErrchk(cudaMemcpy(h_mat, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToHost))
	// multiply_and_print(h_mat,h_inv,n);

	/* Print out inverse matrix */
	printf("Inverse Matrix:\n");
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_inv[x*n + y]);
		}
		printf("\n");
	}

	/* Allocate second matrix on device for multiplication */
	float *d_inv;
	gpuErrchk(cudaMalloc((void **)&d_inv, n*n* sizeof(float)))

	float *d_identity;
	gpuErrchk(cudaMalloc((void **)&d_identity, n*n* sizeof(float)))

	/* Copy inverse matrix to device */
	gpuErrchk(cudaMemcpy(d_inv, h_inv, n*n * sizeof(float), cudaMemcpyHostToDevice))

	/* Multiply matrix with inverse */
	matrix_multiply(d_identity,d_mat,d_inv,n);

	/* Copy random matrix to host */
	gpuErrchk(cudaMemcpy(h_inv, d_identity, n*n * sizeof(float), cudaMemcpyDeviceToHost))
	/* Print out identity matrix */
	printf("Identity Matrix:\n");
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			printf("%1.4f ", h_inv[x*n + y]);
		}
		printf("\n");
	}


	int ur = is_unity_matrix(d_identity,n);
	if(ur)
		printf("SUCCESS! Matix inversion was successfull!!!!!\n");
	else
		printf("FAILED! Matrix inversion not successfull\n");

	free(h_mat);
	cudaFree(d_mat);
	cudaFree(d_inv);
}

static void multiply_and_print(float* A, float* Ainv, int n)
{
	printf("Multiply and print:\n");
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			float sum=0;
			for(int k=0;k<n;k++)
			{
				sum+= A[x*n+k]*Ainv[k*n+y];
			}
			printf("%1.3f",sum);
			if(y!= n-1)
				printf(", ");
		}
		printf("\n");
	}
}
