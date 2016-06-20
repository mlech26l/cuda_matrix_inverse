#ifndef MY_TOOLS_H
#define MY_TOOLS_H


#define cudaCheck(ans) do{if(ans != cudaSuccess){fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(ans),  __FILE__, __LINE__); exit(EXIT_FAILURE);} }while(false)

#define gpuErrchk(ans) { tools_gpuAssert((ans), __FILE__, __LINE__); }
/*
    Debug output
*/
void tools_gpuAssert(cudaError_t code, const char *file, int line);

/*
    Allocates the memory and creates a ID Matrix with n x n dimension
*/
float * tools_create_identity_matrix(int n);

/*
  Print a Matrix with with N x N dimension
*/
void tools_print_matrix(float * matrix, int N);

/*
  Print a Matrix more beautiful 
*/
void tools_WAprint(int size_of_one_side, float * matrix);

/*
  checks for zero with a window of e^-5
*/
int tools_zero(float f);

/*
    Reads a matrix from stin
    returns the size of the matrix on success, otherwise -1
*/
int tools_read_matrix(float *** _matrix);


/*
  simply check the bit patterns.. hope that the gpu uses the same precision as the cpu
*/
int tools_is_equal(float * a, float * b, int size);

#endif