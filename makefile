# Sample makefile

lib = -lcurand
ccopt = -arch sm_30 -ccbin g++ -g
CC=/usr/local/cuda/bin/nvcc

all: bin/matrix.exe

bin/matrix.exe: obj/main.obj obj/random_matrix.obj obj/matrix_multiplication.obj obj/unity_matrix.obj obj/matrix.obj
	$(CC) -o $@ $(lib) $(ccopt) $^

	
obj/unity_matrix.obj: src/unity_matrix.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/random_matrix.obj: src/random_matrix.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj/matrix_multiplication.obj: src/matrix_multiplication.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^
	
obj/main.obj: src/main.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^
	
obj/matrix.obj: src/matrix.cu
	$(CC) -c -o $@ $(lib) $(ccopt) $^
		
	
clean:
	rm obj/*.obj
	rm bin/matrix.exe
