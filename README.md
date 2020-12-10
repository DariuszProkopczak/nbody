# nbody
n-body gravity simulation using the MPI framework

Below video was generated on Blue Gene computer at Argonne lab using 100000 bodies:

![Alt Text](gif/nbody_simulation.gif)

## Dependencies
MPI and OpenMP

## Compilation
(after installing MPI and OpenMP)
mpicc -o nbody_parallel -std=c99 -lm -fopenmp -O3 -Wall nbody_parallel.c

or

make

## To run:
mpirun -n 4 ./nbody_parallel 3000 400 0.2 4

Arguments: &ltnumber of bodies&gt &ltnumber of iterations&gt &ltdt&gt &ltnumber of openmp threads per rank&gt

can also run serial version like this:
	./nbody_serial 3000 400 0.2 4
