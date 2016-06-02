# Sample makefile

lib = -lcurand
ccopt = -O3 -arch sm_20 -ccbin g++

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
