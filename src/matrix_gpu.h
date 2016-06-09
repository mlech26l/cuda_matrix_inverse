/*
 * matrix_gpu.h
 *
 *  Created on: 7 Jun 2016
 *      Author: liten
 */

#ifndef MATRIX_GPU_H_
#define MATRIX_GPU_H_

void inverse_gpu(float * in, int size, float * out, int * success);
__global__ void subtract_row_gpu(float * source, float * target, float scale, int size);
__global__ void divide_row_gpu(float denominator,float * vector, int start_idx, int size);



#endif /* MATRIX_GPU_H_ */
