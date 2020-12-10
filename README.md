# nbody
n-body gravity simulation using the MPI framework
Below video was generated on Blue Gene computer at Argonne lab using 100000 bodies:

![Alt Text](gif/nbody_simulation.gif)

# dependencies
MPI and OpenMP

# compilation
(after installing MPI and OpenMP)
mpicc -o nbody_parallel -std=c99 -lm -fopenmp -O3 -Wall nbody_parallel.c

or

make

# to run
Example:

mpirun -n 4 ./nbody_parallel 3000 400 0.2 4

The first argument is number of bodies.
The second is number of iterations.
The third is dt.
The fourth is number of openmp threads per rank.

can also run serial version like this:
	./nbody_serial 3000 400 0.2 4
