# Sample makefile

lib = -lcurand
ccopt = -O3 -arch sm_30 -ccbin g++ -g
CC=/usr/local/cuda/bin/nvcc

OBJs= obj/wingetopt.obj obj/device_query.obj obj/pivoting.obj obj/main.obj obj/tools.obj obj/random_matrix.obj obj/gauss.obj obj/lu_dec.obj obj/matrix_multiplication.obj obj/identity_matrix.obj obj/test.obj 

all: bin/matrix.exe

bin/matrix.exe: $(OBJs) | bin
	$(CC) -o $@ $(lib) $(ccopt) $^

obj/%.obj: src/%.cu | obj
	$(CC) -c -o $@ $(lib) $(ccopt) $^

obj:
	mkdir -p obj

bin:
	mkdir -p bin
	
clean:
	rm -r obj
	rm -r bin
