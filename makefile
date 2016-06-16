# Sample makefile

lib = -lcurand
ccopt = -arch sm_30 -ccbin g++ -g
CC=/usr/local/cuda/bin/nvcc

all: bin/matrix.exe

bin/matrix.exe: obj/main.obj obj/testing_util.obj obj/random_matrix.obj obj/matrix_multiplication.obj  obj/device_query.obj obj/gpu_enabled.obj obj/gpu_pivoting.obj obj/identity_matrix.obj obj/matrix.obj obj/matrix_gpu.obj
	$(CC) -o $@ $(lib) $(ccopt) $^


obj/unity_matrix.obj: src/unity_matrix.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/testing_util.obj: src/testing_util.cu
		$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/random_matrix.obj: src/random_matrix.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/device_query.obj: src/device_query.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/gpu_pivoting.obj: src/gpu_pivoting.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/gpu_enabled.obj: src/gpu_enabled.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/matrix_multiplication.obj: src/matrix_multiplication.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/main.obj: src/main.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/matrix.obj: src/matrix.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/matrix_gpu.obj: src/matrix_gpu.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/identity_matrix.obj: src/identity_matrix.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^
	
clean:
	rm obj/*.obj
	rm bin/matrix.exe
