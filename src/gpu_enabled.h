#ifndef GPU_ENABLED_H_
#define GPU_ENABLED_H_

int lup_matrix_inverse_gpu(float *A, float *Ainv, int n);
int lup_matrix_inverse_cpu(float *A, float *Ainv, int n);
int lup_decomposition_gpu(float *A, int *pivot, int n);
int lup_decomposition_cpu(float *A, int *pivot, int n);
void lup_invert_column_gpu(float *LU,  int *pivot, float *d_A1, int n) ;
void lup_invert_column_cpu(float *LU, int col, int* pivot, float* x, int n) ;

#endif
