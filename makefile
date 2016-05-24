# Sample makefile

lib = -lcurand
ccopt = -O3 -arch sm_20 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin"

all: bin/matrix.exe

bin/matrix.exe: obj/unity_matrix.obj obj/random_matrix.obj obj/main.obj obj/matrix_multiplication.obj
	nvcc -o $@ $(lib) $(ccopt) $^

	
obj/unity_matrix.obj: src/unity_matrix.cu
	nvcc -c -o $@ $(lib) $(ccopt) $^

obj/random_matrix.obj: src/random_matrix.cu
	nvcc -c -o $@ $(lib) $(ccopt) $^

obj/matrix_multiplication.obj: src/matrix_multiplication.cu
	nvcc -c -o $@ $(lib) $(ccopt) $^
	
obj/main.obj: src/main.cu
	nvcc -c -o $@ $(lib) $(ccopt) $^
		
	
clean:
	rm obj/*.obj
	rm bin/matrix.exe
	
# nvcc -c -o bin/random_matrix.obj -lcurand -O3 -arch sm_20 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" src/random_matrix.cu
# nvcc -c -o bin/matrix_util.obj -lcurand -O3 -arch sm_20 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" src/matrix_util.cu
# nvcc -c -o bin/main.obj -lcurand -O3 -arch sm_20 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" src/main.cu
# nvcc -o bin/matrix.exe -lcurand -O3 -arch sm_20 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" bin/main.obj bin/matrix_util.obj bin/random_matrix.obj
# "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\LINK.exe"  bin/matrix_util.obj bin/random_matrix.obj bin/main.obj