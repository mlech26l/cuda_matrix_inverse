# include <stdio.h>
# include <float.h>
# include <stdlib.h>
# include <math.h>



/* This function performs LUP decomposition of the to-be-inverted matrix 'A'. It is
 * defined after the function 'main()'.
 * * */
//static int LUPdecompose(int size, float A[size][size], int P[size]);
static int LUPdecompose(int size, float* A, int* P);
/* This function calculates inverse of the matrix A. It accepts the LUP decomposed
 * matrix through 'LU' and the corresponding pivot through 'P'. The inverse is
 * returned through 'LU' itself. The spaces 'B', 'X', and 'Y' are used temporary,
 * merely to facilitate the computation. This function is defined after the function
 * 'LUPdecompose()'.
 * * */
 //static int LUPinverse(int size, int* P, float LU[size][size],\
   //                   float B[size][size], float X[size], float Y[size]);
static int LUPinverse(int size, int* P, float* LU,\
                      float* B, float* X, float* Y);


int matrix_inverse_host_lup(float* A, int n)
{
	float* C= (float*)malloc(sizeof(float)*(n+1)*(n+1));
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			C[(x+1)*n+y+1] = A[x*n + y];
		}
		printf("\n");
	} 
	
	int* P= (int*)malloc(sizeof(int)*(n+1));

	float* X= (float*)malloc(sizeof(float)*(n+1));
	float* Y= (float*)malloc(sizeof(float)*(n+1));
	float* B= (float*)malloc(sizeof(float)*(n+1)*(n+1));
	
	if(LUPdecompose(n+1, C, P) < 0) return -1;
	if(LUPinverse(n+1, P, C, B, X, Y) < 0) return -1;
	
	for(int x = 0; x < n; x++) {
		for(int y = 0; y < n; y++) {
			 A[x*n + y] = C[(x+1)*n+y+1];
		}
		printf("\n");
	} 
	
	free(C);
	free(P);
	free(X);
	free(Y);
	free(B);
	
	return 0;
}



/* This function decomposes the matrix 'A' into L, U, and P. If successful,
 * the L and the U are stored in 'A', and information about the pivot in 'P'.
 * The diagonal elements of 'L' are all 1, and therefore they are not stored. */
static int LUPdecompose(int size, float* A, int* P)
   {
    int i, j, k, kd = 0, T;
    float p, t;

 /* Finding the pivot of the LUP decomposition. */
    for(i=1; i<size; i++) P[i] = i; //Initializing.

    for(k=1; k<size-1; k++)
       {
        p = 0;
        for(i=k; i<size; i++)
           {
            t = A[i*size+k];
            if(t < 0) t *= -1; //Abosolute value of 't'.
            if(t > p)
                {
                 p = t;
                 kd = i;
                }
           }

        if(p == 0)
           {
            printf("\nLUPdecompose(): ERROR: A singular matrix is supplied.\n"\
                   "\tRefusing to proceed any further.\n");
            return -1;
           }

     /* Exchanging the rows according to the pivot determined above. */
        T = P[kd];
        P[kd] = P[k];
        P[k] = T;
        for(i=1; i<size; i++)
            {
             t = A[kd*size+i];
             A[kd*size+i] = A[k*size+i];
             A[k*size+i] = t;
            }

        for(i=k+1; i<size; i++) //Performing substraction to decompose A as LU.
            {
             A[i*size+k] = A[i*size+k]/A[k*size+k];
             for(j=k+1; j<size; j++) A[i*size+j] -= A[i*size+k]*A[k*size+j];
            }
        } //Now, 'A' contains the L (without the diagonal elements, which are all 1)
          //and the U.

    return 0;
   }




/* This function calculates the inverse of the LUP decomposed matrix 'LU' and pivoting
 * information stored in 'P'. The inverse is returned through the matrix 'LU' itselt.
 * 'B', X', and 'Y' are used as temporary spaces. */
static int LUPinverse(int size, int* P, float* LU,\
                      float* B, float* X, float* Y)
   {
    int i, j, n, m;
    float t;

  //Initializing X and Y.
    for(n=1; n<size; n++) X[n] = Y[n] = 0;

 /* Solving LUX = Pe, in order to calculate the inverse of 'A'. Here, 'e' is a column
  * vector of the identity matrix of size 'size-1'. Solving for all 'e'. */
    for(i=1; i<size; i++)
     {
    //Storing elements of the i-th column of the identity matrix in i-th row of 'B'.
      for(j = 1; j<size; j++) B[i*size+j] = 0;
      B[i*size+i] = 1;

   //Solving Ly = Pb.
     for(n=1; n<size; n++)
       {
        t = 0;
        for(m=1; m<=n-1; m++) t += LU[n*size+m]*Y[m];
        Y[n] = B[i*size+P[n]]-t;
       }

   //Solving Ux = y.
     for(n=size-1; n>=1; n--)
       {
        t = 0;
        for(m = n+1; m < size; m++) t += LU[n*size+m]*X[m];
        X[n] = (Y[n]-t)/LU[n*size+n];
       }//Now, X contains the solution.

      for(j = 1; j<size; j++) B[i*size+j] = X[j]; //Copying 'X' into the same row of 'B'.
     } //Now, 'B' the transpose of the inverse of 'A'.

 /* Copying transpose of 'B' into 'LU', which would the inverse of 'A'. */
    for(i=1; i<size; i++) for(j=1; j<size; j++) LU[i*size+j] = B[j*size+i];

    return 0;
   }
