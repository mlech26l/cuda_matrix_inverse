#ifndef MY_GAUSS_H
#define MY_GAUSS_H

/*
  takes a matrix and returns a pointer to a inverse matrix if it exists
  the input is modified, if no inverse is found it returns a null pointer
  out needs to point to a identity matrix
*/
int gauss_inverse_cpu(float * in, int size, float * out);

int gauss_inverse_gpu(float * in, int size, float * out);


#endif