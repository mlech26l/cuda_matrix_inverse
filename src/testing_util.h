#ifndef TESTING_UTIL_H_
#define TESTING_UTIL_H_

void test_matrix_mathias_functions(void);
__global__ void print_matrix_on_device_kernel(float* d_A, int n);

#endif
