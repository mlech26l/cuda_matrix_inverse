#ifndef IDENTITY_MATRIX_H_
#define IDENTITY_MATRIX_H_

// Creates a identity matrix on the device
float* get_dev_identity_matrix(int n);

// Checks if the matrix of dimension n*n is the identity matrix,
// uses the reduction technique as discussed in the lecture.
// Reduction template from \CUDA Samples\v7.5\6_Advanced\reduction
int is_identity_matrix(float* d_mat, int n);

#endif
