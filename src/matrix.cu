#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

//returns the size of the matrix on success, otherwise -1
int read_matrix(float *** _matrix){
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



//takes a matrix and returns a pointer to a invers matrix if it exists
//the input is modified, if no inverse is found it returns a null pointer
//out needs to point to a identity matrix
void inverse_cpu(float * in, int size, float * out, int * success){

	//Gaussian elimination step
	int i;
	for(i = 0; i < size; i++){
		int ret;
		zero(in[i*size + i], &ret);
		if(ret){ // "equals" zero
			find_and_swap_up_row(in, i, size, &ret);
			if(ret == 0){
				*success = 0;
				return;
			}
			find_and_swap_up_row(out, i, size, &ret);
		}
		// here we have something in in[i][i] to work with
		
		//scale the row so that [i][i] == 1
		divide_row(in[i*size + i], &out[i*size], size);
		divide_row(in[i*size + i], &in[i*size], size);
		
		//zero out the column below
		int j;
		for(j = i + 1; j < size; j++){
			subtract_row(&out[i*size], &out[j*size], in[j*size + i], size);
			subtract_row(&in[i*size], &in[j*size], in[j*size + i], size); // in row, out/target row, scale the in row, size
			
		}
	}
	
	
	//back substitution step
	int column;
	for(column = size - 1; column >= 1; column--){
		int row;
		for(row = column-1; row >= 0; row--){
			float factor = in[row*size + column];

			int j;
			for(j = 0; j < size; j++){
				out[row*size + j] = out[row*size + j] - factor*out[column*size +j];
				in[row*size + j] = in[row*size + j] - factor*in[column*size +j];
			}			
		}
	}
	
	*success = 1;
}


__host__ __device__
void zero(float f, int * ret){
	if(abs(f*1e5) < 1){
		*ret = 1;
	}
	else {
		*ret = 0;
	}
}

// in row, out/target row, scale the in row, size
__host__ __device__
void subtract_row(float * source, float * target, float scale, int size){
	int i;
	for(i = 0;i < size; i++){
		target[i] = target[i] - (source[i] * scale);
	}
}

void divide_row(float denominator,float * row, int size){
	int i;
	for(i = 0;i < size; i++){
		row[i] = row[i]/denominator;
	}
}


__host__ __device__
void divide_row_gpu(float denominator,float * row, int size){
	int i;
	for(i = 0;i < size; i++){
		row[i] = row[i]/denominator;
	}
}



void find_and_swap_up_row(float * a, int row, int size, int * ret){
	int i;
	for(i = (row + 1)*size;i < size*size; i++){
		
		int * ret = 0;
		//swap rows
		zero(a[row*size + i],ret);
		if(*ret == 0){

			//swap the elements of the rows, one element at the time.
			int j;
			for(j = i;j < i + size; j++){
				float temp = a[j];
				a[j] = a[row*size];
				a[row*size] = temp;
			}

			*ret = 1;
			return;
		}
	}
	
	*ret = 0;;
}



void print_matrix(float * matrix, int N){
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

