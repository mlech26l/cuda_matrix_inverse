#ifndef TESTING_UTIL_H_
#define TESTING_UTIL_H_



__global__ void print_matrix_on_device_kernel(float* d_A, int n);

void test_matrix_mathias_functions(int n);


#endif
