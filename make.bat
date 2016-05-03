nvcc -o bin/main.exe -lcurand -O3  -arch sm_20 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" src\main.cu src/random_matrix.cu
