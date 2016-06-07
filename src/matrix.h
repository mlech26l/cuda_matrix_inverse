#ifndef MY_MATRIX 
#define MY_MATRIX

int read_matrix(float *** _matrix);
void print_matrix(float * matrix, int N);
void free_matrix(float * matrix, int N);
__host__ __device__ void inverse(float * in, int size, float * out, int * success);
__host__ __device__ void subtract_row(float * source, float * target, float scale, int size);
__host__ __device__ void divide_row(float denominator,float * row, int size);
__host__ __device__ void find_and_swap_up_row(float *, int row, int size, int * ret);
__host__ __device__ void zero(float,int * ret);

#endif
