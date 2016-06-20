#ifndef RANDOM_MATRIX_H_
#define RANDOM_MATRIX_H_


/* Allocates an array of size n-by-n on the device
 * and initializies it with random variables.
 * The random variables are in the range of (0, max]
 * If truncate != 0 the digits after the decimal point are truncated
 * i.e. instead of 5.38463 the variable will be 5.0000 
 */
float* random_matrix_generate(int n, float max, int truncate);

#endif