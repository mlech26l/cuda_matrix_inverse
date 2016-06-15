/*
 * matrix_gpu.h
 *
 *  Created on: 7 Jun 2016
 *      Author: liten
 */

#ifndef MATRIX_GPU_H_
#define MATRIX_GPU_H_

void inverse_gpu(float * in, int size, float * out, int * success);
__global__ void divide_2rows_gpu(int denominator_idx, float * denom_src_vec, float * vector, float * vector2, int size);
__global__ void subtract_rows_gpu(int i, float * in, float * out, int size);
__global__ void zero_out_column_gpu(int column, int direction, float * in, float * out, int size);

__global__ void divide_row_gpu(int denominator_idx, float * denom_src_vec, float * vector, int start_idx, int size);
__global__ void subtract_row_gpu(float * source, float * target, float scale, int size);


#endif /* MATRIX_GPU_H_ */