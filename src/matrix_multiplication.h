#ifndef MATRIX_MULTIPLICATION_H_
#define MATRIX_MULTIPLICATION_H_

#define MULTIPLY_BLOCK_SIZE 3

// Multiplies A and B, uses the cache optimized algorithm as discussed in the lecture
// Kernel code taken from \CUDA Samples\v7.5\0_Simple\matrixMul
void matrix_multiply(float* C, float* A, float* B, int n);


#endif
