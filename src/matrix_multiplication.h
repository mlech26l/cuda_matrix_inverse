#ifndef MATRIX_MULTIPLICATION_H_
#define MATRIX_MULTIPLICATION_H_

#define MULTIPLY_BLOCK_SIZE 16

// Multiplies A and B, uses the cache optimized algorithm as discussed in the lecture
void matrix_multiply(float* C, float* A, float* B, int n);


#endif