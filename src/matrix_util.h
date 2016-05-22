#ifndef MATRIX_UTIL_H_
#define MATRIX_UTIL_H_

// Multiplies A and B, uses the cache optimized algorithm as discussed in the lecture
void mat_mul_dev( float* C, float* A, float* B, int N);

// Creates a unity matrix on the device
float* get_dev_unity_matrix(int n);

int is_unity_matrix(float* d_mat, int n);

#endif