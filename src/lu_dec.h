#ifndef MY_LU_DEC_H
#define MY_LU_DEC_H


int lu_dec_matrix_inverse_cpu(float *A, float* A1,int n);

int lu_dec_matrix_inverse_gpu(float *d_A, float* d_A1,int n);

#endif