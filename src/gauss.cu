/* Matrix Inversion 
 * Group F: M. Lechner, P. Knöbel, J. Lövhall
 *
 * Gauss-Jordan implementation of Matrix Inversion  
*/

#include "includes.h"

static int find_and_swap_up_row(float * a, int row, int size){
	int i;
	for(i = (row + 1)*size;i < size*size; i++){
		//swap rows
		;
		if(tools_zero(a[row*size + i])){

			//swap the elements of the rows, one element at the time.
			int j;
			for(j = i;j < i + size; j++){
				float temp = a[j];
				a[j] = a[row*size];
				a[row*size] = temp;
			}

			return 1;
		}
	}
	
	return 0;;
}

static void divide_row(float denominator,float * row, int size){
	int i;
	for(i = 0;i < size; i++){
		row[i] = row[i]/denominator;
	}
}

// in row, out/target row, scale the in row, size
static void subtract_row(float * source, float * target, float scale, int size){
	int i;
	for(i = 0;i < size; i++){
		target[i] = target[i] - (source[i] * scale);
	}
}

/*
  takes a matrix and returns a pointer to a inverse matrix if it exists
  the input is modified, if no inverse is found it returns a null pointer
  out needs to point to a identity matrix
*/
int gauss_inverse_cpu(float * in, int size, float * out){

	//Gaussian elimination step
	int i;
	for(i = 0; i < size; i++){
		if(tools_zero(in[i*size + i])){ 
      // "equals" zero
			if(find_and_swap_up_row(in, i, size)){
        //inverse matrix does not exists
				return 0;
			}
			find_and_swap_up_row(out, i, size);
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
	
	return 1;
}



static __global__
void zero_out_column_gpu(int column, int direction, float * in, float * out, int size){
	int idx = blockIdx.x*blockDim.x  + threadIdx.x;
	if(idx < size){
		int j;
		for(j = column + direction; j < size && j >= 0; j+= direction){
			float scale = in[j*size + column];

			__syncthreads();
			out[idx + j*size] = out[idx+j*size] - (out[idx + column*size] * scale);
			__syncthreads();
			in[idx + j*size] = in[idx+j*size] - (in[idx + column*size] * scale);
			__syncthreads();
		}
	}
}

//takes vector[denominator_idx] as index and divides all elements in the row from vector and vector2
static __global__
void divide_2rows_gpu(int denominator_idx, float * denom_src_vec, float * vector, float * vector2, int size){
	int idx = blockIdx.x*blockDim.x  + threadIdx.x;
	float denominator = denom_src_vec[denominator_idx];

	__syncthreads();

	if(idx < size){
		vector[idx] = vector[idx]/denominator;
		__syncthreads();
		vector2[idx] = vector2[idx]/denominator;
		__syncthreads();
	}
}


int gauss_inverse_gpu(float * in, int size, float * out){

	float * d_in;
	float * d_out;
	gpuErrchk(cudaMalloc((void **)&d_in, size*size* sizeof(float)))
	gpuErrchk(cudaMalloc((void **)&d_out, size*size* sizeof(float)))

	gpuErrchk(cudaMemcpy(d_in, in, size*size*sizeof(float), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(d_out, out, size*size*sizeof(float), cudaMemcpyHostToDevice))
	cudaDeviceSynchronize();
	//Gaussian elimination step
	int i;
	for(i = 0; i < size; i++){
		//todo, there was a check for the possibility to invert here before (row swap), should be brought back adventually

		//scale the row so that [i][i] == 1
		divide_2rows_gpu<<<size/32 + 1,32>>>(i*size + i,d_in , d_in + i*size, d_out + i*size, size);
		cudaDeviceSynchronize();
		//zero out the column below
		zero_out_column_gpu<<<size/32 + 1,32>>>(i, 1, d_in, d_out, size);
		cudaDeviceSynchronize();
	}

	//back substitution step
	int column;
	for(column = size - 1; column >= 1; column--){
		/*if(column == 11){
			gpuErrchk(cudaMemcpy(out, d_out, size*size*sizeof(float), cudaMemcpyDeviceToHost))
			cudaDeviceSynchronize();
			printf("error after this step\n");
			print_matrix(out, size);
			printf("\n\n");
		}*/
		zero_out_column_gpu<<<size/32 + 1,32>>>(column, -1, d_in, d_out, size);
		cudaDeviceSynchronize();
		/*if(column == 11){
			gpuErrchk(cudaMemcpy(out, d_out, size*size*sizeof(float), cudaMemcpyDeviceToHost))
			cudaDeviceSynchronize();
			printf("error before this step\n");
			print_matrix(out, size);
			printf("\n\n");
		}*/
	}

	//get the inverted matrix back to host memory
	gpuErrchk(cudaMemcpy(out, d_out, size*size*sizeof(float), cudaMemcpyDeviceToHost))

	return 1;
}

