#ifndef UNITY_MATRIX_H_
#define UNITY_MATRIX_H_

// The cuda code in this module is based on the templates in CUDA Samples\v7.5\6_Advanced\reduction

// Creates a unity matrix on the device
float* get_dev_unity_matrix(int n);

// Checks if the matrix of dimension n*n is the unit matrix
int is_unity_matrix(float* d_mat, int n);

#endif