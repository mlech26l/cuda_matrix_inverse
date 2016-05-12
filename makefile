# Sample makefile

src = src\main.cu src/random_matrix.cu
lib = -lcurand
ccopt = -O3 -arch sm_20 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin"

all: bin/matrix.exe

bin/matrix.exe: $(src)
	nvcc -o $@ $(lib) $(ccopt) $(src)
