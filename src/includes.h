/* Matrix Inversion 
 * Group F: M. Lechner, P. Knöbel, J. Lövhall
 *
 * All necessary include files
*/

// Standard libraries 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//Cuda libraries
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

  
//Fixes for windows CC
#include "wingetopt.h"


//Src libraries
#include "pivoting.h"
#include "device_query.h"
#include "tools.h"
#include "random_matrix.h"
#include "matrix_multiplication.h"
#include "identity_matrix.h"

#include "gauss.h"
#include "lu_dec.h"

#include "test.h"


