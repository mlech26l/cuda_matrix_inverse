/* Matrix Inversion 
 * Group F: M. Lechner, P. Knöbel, J. Lövhall
 *
 * Tools used for debugging printing and more
*/

#include "includes.h"

/*
    Debug output
*/
void tools_gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

/*
    Allocates the memory and creates a ID Matrix with n x n dimension
*/
float * tools_create_identity_matrix(int n){
	float * out = (float *)malloc(sizeof(float)*n*n);
	int i;
	for(i = 0; i < n; i++){
		int j;
		for(j = 0;j < n; j++){
			out[i*n + j] = 0.0;
		}
		out[i*n + i] = 1.0;
	}
	return out;
}


/*
  Print a Matrix with with N x N dimension
*/
void tools_print_matrix(float * matrix, int N){
	int i;
	for(i = 0; i < N; ++i){
		int j;
		for(j = 0;j < N; ++j){
			printf("%f ",matrix[i*N + j]);
		}
		printf("\n\r");
	}
	printf("\n\r");
}

/*
  Print a Matrix more beautiful 
*/
void tools_WAprint(int size_of_one_side, float * matrix){
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

/*
  checks for zero with a window of e^-5
*/
int tools_zero(float f){
	if(abs(f*1e5) < 1){
		return 1;
	}
  return 0;
}


/*
    Reads a matrix from stin
    returns the size of the matrix on success, otherwise -1
*/
int tools_read_matrix(float *** _matrix){
	int N;
	int r = scanf("%d",&N);
	
	if(r== EOF){
		printf("error reading input\n\rexiting");
		return -1;
	}else if (r == 0){
		printf("no input, please supply a matrix to stdin");
		return -1;
	}
	//we read the integer
	
	//create arrays to store the matrix in	
	float ** matrix = (float **)malloc((sizeof(float*)*N)); //square matrix N*N
	
	int i;
	for(i = 0; i < N; ++i){
		matrix[i] = (float*)malloc(sizeof(float)*N);
	}	
	
	for(i = 0; i < N; ++i){
		int j;
		for(j = 0;j < N; ++j){
			int r;
			int t = 0; //tries
			while((r = scanf("%f",&(matrix[i][j]))) == 0 && t < 100){
				t++;
			}
			if(r == EOF){
				printf("error reading input on row %d and column %d \n\rexiting",i,j);
			//	free_matrix(matrix,N);
				return -1;
			} else if(r == 0){
				printf("failed to read input after multiple tries\n\r");
				//free_matrix(matrix,N);
				return -1;
			}
		}
	}
	*_matrix = matrix;
	return N;
}

/*
  simply check the bit patterns.. hope that the gpu uses the same precision as the cpu
*/
int tools_is_equal(float * a, float * b, int size){
	int i;
	int ret = 1;
	for(i = 0;i < size;i++){
		if(abs(a[i] - b[i]) > 0.00001){
			printf("element %d is not equal. GPU = %f, CPU = %f\n",i,a[i],b[i]);
			ret = 0;
		}
	}
	return ret;
}













