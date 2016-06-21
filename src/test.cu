/* Matrix Inversion 
 * Group F: M. Lechner, P. Knöbel, J. Lövhall
 *
 * All Test suites
*/

#include "includes.h"


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
  	matrix_multiplication(d_identity,d_mat2,d_inv,n);
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
  	int ur = identity_matrix(d_identity,n);
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
  	matrix_multiplication(d_identity,d_mat2,d_inv,n);
    printf("[OK]\n");

    // printf("Identity matrix:\n");
    // print_matrix_on_device_kernel<<<1,1>>>(d_identity,n);

    printf("Check host matrix for Identity...");
  	ur = identity_matrix(d_identity,n);
    printf("[OK]\n");
  	if(!ur)
  		printf("Host matrix inversion failed!\n");
}

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

void test_gauss(int n){
	printf("running Jakobs tests.\n");

  float time = 0;
  cudaEvent_t start, stop;

	float *matrix;
	float * matrix_org;
	printf("\nDoing matrix inversion test with n=%d\n",n);

	if(n == 3){
		matrix = tools_create_identity_matrix(n);//(float *)malloc(n*n* sizeof(float));
		matrix[1] = 1;
		matrix[6] = 1;
		matrix_org = tools_create_identity_matrix(n); //used instead of malloc because lazy and easy..
	} else {
		matrix = (float *)malloc(sizeof(float)*n*n);
		matrix_org = (float *)malloc(sizeof(float)*n*n);
		float * d_mat;
		d_mat = random_matrix_generate(n,100,1);
		gpuErrchk(cudaMemcpy(matrix, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToHost))
    cudaCheck(cudaFree(d_mat));
	} 

	int i;
	for(i = 0;i <n*n; i++){
		matrix_org[i] = matrix[i];
	}
	float* inverse = tools_create_identity_matrix(n);
	float* inverse_matrix_cpu = tools_create_identity_matrix(n);

	/* Print out test matrix */
	if(n == 3){
		printf("test Matrix org:\n");
		tools_print_matrix(matrix,n);

		printf("test Matrix:\n");
		tools_print_matrix(matrix,n);
		tools_WAprint(n,matrix);
	}

  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaEventSynchronize(start);

  gauss_inverse_gpu(matrix, n, inverse);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("CUDA inverse took ms: %f\n", time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaEventSynchronize(start);

  //running cpu test first because it has singularity check.
  //inversion destroys the matrix
  int succ= gauss_inverse_cpu(matrix, n, inverse_matrix_cpu);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("CPU inverse took ms: %f\n", time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  
	if(!succ)
	{
		printf("Matrix singular!");
		exit(EXIT_SUCCESS);
	}

	//restore the matrix
	for(i = 0;i <n*n; i++){
		matrix[i] = matrix_org[i];
	}




	if(!tools_is_equal(inverse,inverse_matrix_cpu,n*n)){
		printf("matrixes not equal. printing.\n\n");
		printf("gpu matrix \n");

		tools_print_matrix(inverse, n);
		printf("\n\ncpu matrix \n");
		tools_print_matrix(inverse_matrix_cpu, n);

		printf("start matrix\n");
		tools_WAprint(n,matrix_org);
	} else {
		printf("matrixes equal. all is good. \n");
	}


	free(matrix);
	free(matrix_org);
	free(inverse);
	free(inverse_matrix_cpu);
}

void test_cofactors(int n){
  /*
  float *d_mat, h_mat; 
  float *d_inv, h_inv; 
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	d_mat = random_matrix_generate(n,100,1);
  cudaCheck(cudaMalloc((void**)&d_inv, sizeof(float)*n*n));

  h_mat=(float*)malloc(sizeof(float)*n*n);
  h_inv=(float*)malloc(sizeof(float)*n*n);
  cudaCheck(cudaMemcpy(h_mat, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToHost));

  if(n==5){
    
  }
  */
  
  printf("Not implemented!");
}

void test_lu_decomposition(int n)
{
  /* 100MB Matrix (5000x5000) take 4min to invert on GPU (GT 525m, with 96 threads)
  vs 28 min on a Intel i5 2430m core (= 7 x speedup)
  */
  printf("Mathias test utility!\n");
  //test_gpu_pivoting();
	float *d_mat, *d_inv, *d_mat2;
	float *h_mat, *h_inv;

	printf("\nDoing LU matrix inversion test with n=%d\n",n);


	d_mat = random_matrix_generate(n,100,1);
  cudaCheck(cudaMalloc((void**)&d_inv, sizeof(float)*n*n));
  cudaCheck(cudaMalloc((void**)&d_mat2, sizeof(float)*n*n));

  h_mat=(float*)malloc(sizeof(float)*n*n);
  h_inv=(float*)malloc(sizeof(float)*n*n);
  cudaCheck(cudaMemcpy(h_mat, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToHost));

	/* Copy random matrix on device */
	cudaCheck(cudaMemcpy(d_mat2, d_mat, n*n * sizeof(float), cudaMemcpyDeviceToDevice));


  if(lu_dec_matrix_inverse_gpu(d_mat, d_inv, n)==0)
	{
		printf("Matrix singular!");
		exit(EXIT_SUCCESS);
	}

  lu_dec_matrix_inverse_cpu(h_mat,h_inv,n);

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


