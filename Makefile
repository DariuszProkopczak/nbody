all: nbody_parallel nbody_serial

nbody_parallel: nbody_parallel.c
	mpicc -o nbody_parallel -std=c99 -lm -fopenmp -O3 -Wall nbody_parallel.c

nbody_serial: nbody_serial.c
	gcc -o nbody_serial -std=c99 -lm -fopenmp -O3 -Wall nbody_serial.c
