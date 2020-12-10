#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

// Code is built off of https://github.com/jtramm/nbody_serial

typedef struct {
	// Location r_i = (x,y,z)
	double x;
	double y;
	double z;
	// Velocity v_i = (vx, vy, vz)
	double vx;
	double vy;
	double vz;
	// Mass
	double mass;
} Body;

void print_bodies(Body * bodies, int bodies_per_rank, int mype, int iteration)
{

	for (int i=0; i<bodies_per_rank; i++)
		printf("Rank %d, iteration %d, body %d, (x, y, z): %f, %f, %f, (vx, vy, vz): %f, %f, %f mass: %f\n", mype, iteration, i, bodies[i].x, bodies[i].y, bodies[i].z,  bodies[i].vx, bodies[i].vy, bodies[i].vz, bodies[i].mass);

}

// A 63-bit LCG
// Returns a double precision value from a uniform distribution
// between 0.0 and 1.0 using a caller-owned state variable.
double LCG_random_double(uint64_t * seed)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}

// "Fast Forwards" an LCG PRNG stream
// seed: starting seed
// n: number of iterations (samples) to forward
// Returns: forwarded seed value
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;

	n = n % m;

	uint64_t a_new = 1;
	uint64_t c_new = 0;

	while(n > 0)
	{
		if(n & 1)
		{
			a_new *= a;
			c_new = c_new * a + c;
		}
		c *= (a + 1);
		a *= a;

		n >>= 1;
	}
	return (a_new * seed + c_new) % m;
}

// will place each particle in one of 8 rectangular prisms
// each prism will have similar velocity
void parallel_randomizeBodies(Body * bodies, int bodies_per_rank, int nBodies, int mype, int nprocs)
{

	uint64_t seed = 42;

	// velocity scaling term
	double vm = 1.0e-2;

	double rect_val;
	double x_rect_offset, y_rect_offset, z_rect_offset;
	double x_vel_direction, y_vel_direction;

	for (int i = 0; i < bodies_per_rank; i++)
	{
		// Fast forward seed to this particle's location in the global PRNG stream.
		// We forward 8 x particle_id, as each particle requires 8 PRNG samples.
		uint64_t particle_seed = fast_forward_LCG(seed, 8*(i+(bodies_per_rank*mype)));

		// will first determine which of 8  boxes to place particle in
		rect_val = LCG_random_double(&particle_seed);

		if (rect_val < 0.125)
		{
			x_rect_offset = -1.0;
			y_rect_offset = -1.0;
			z_rect_offset = -1.0;
			x_vel_direction = 1.0;
			y_vel_direction = 0;
		}
		else if (rect_val < 0.25)
		{
			x_rect_offset = -1.0;
			y_rect_offset = -1.0;
			z_rect_offset = 1.0;
			x_vel_direction = -1.0;
			y_vel_direction = 0.0;
		}
		else if (rect_val < 0.375)
		{
			x_rect_offset = -1.0;
			y_rect_offset = 1.0;
			z_rect_offset = -1.0;
			x_vel_direction = 0.0;
			y_vel_direction = -1.0;
		}
		else if (rect_val < 0.5)
		{
			x_rect_offset = -1.0;
			y_rect_offset = 1.0;
			z_rect_offset = 1.0;
			x_vel_direction = 0.0;
			y_vel_direction = 1.0;
		}
		else if (rect_val < 0.625)
		{
			x_rect_offset = 1.0;
			y_rect_offset = -1.0;
			z_rect_offset = -1.0;
			x_vel_direction = 1.0;
			y_vel_direction = 0.0;
		}
		else if (rect_val < 0.75)
		{
			x_rect_offset = 1.0;
			y_rect_offset = -1.0;
			z_rect_offset = 1.0;
			x_vel_direction = 0.0;
			y_vel_direction = -1.0;
		}
		else if (rect_val < 0.875)
		{
			x_rect_offset = 1.0;
			y_rect_offset = 1.0;
			z_rect_offset = -1.0;
			x_vel_direction = -1.0;
			y_vel_direction = 0.0;
		}
		else 
		{
			x_rect_offset = 1.0;
			y_rect_offset = 1.0;
			z_rect_offset = 1.0;
			x_vel_direction = 1.0;
			y_vel_direction = 0.0;
		}

		// Initialize positions
		bodies[i].x =  LCG_random_double(&particle_seed) + x_rect_offset;
		bodies[i].y =  LCG_random_double(&particle_seed) + y_rect_offset;
		bodies[i].z =  LCG_random_double(&particle_seed) + z_rect_offset;

		// Intialize velocities
		bodies[i].vx =  -1 * x_vel_direction * 2.0 * vm * LCG_random_double(&particle_seed) - vm;
		bodies[i].vy =  -1 * y_vel_direction * 2.0 * vm * LCG_random_double(&particle_seed) - vm;
		bodies[i].vz = (2.0 * vm * LCG_random_double(&particle_seed) - vm) * 0.1;

		// Initialize masses so that total mass of system is constant
		// regardless of how many bodies are simulated.
		bodies[i].mass = LCG_random_double(&particle_seed) / nBodies;
	}
}

void compute_forces_multi_set(Body * local_bodies, Body * remote_bodies, double dt, int n, int mype)
{

	double G = 6.67259e-3;
	double softening = 1.0e-5;

	// For each particle in the set
	int i, j;
	double Fx, Fy, Fz, dx, dy, dz, distance, distance_cubed, m_j, mGd;
	#pragma omp parallel for default(none) private( i, j, Fx, Fy, Fz, dx, dy, dz, distance, distance_cubed, m_j, mGd) shared(n, dt, remote_bodies, local_bodies, G, softening) schedule(static)
	for (i = 0; i < n; i++)
	{ 
		Fx = 0.0;
		Fy = 0.0;
		Fz = 0.0;

		// Compute force from all other particles in the set
		for (j = 0; j < n; j++)
		{
			// F_ij = G * [ (m_i * m_j) / distance^3 ] * (location_j - location_i) 

			// First, compute the "location_j - location_i" values for each dimension
			dx = remote_bodies[j].x - local_bodies[i].x;
			dy = remote_bodies[j].y - local_bodies[i].y;
			dz = remote_bodies[j].z - local_bodies[i].z;

			// Then, compute the distance^3 value
			// We will also include a "softening" term to prevent near infinite forces
			// for particles that come very close to each other (helps with stability)

			// distance = sqrt( dx^2 + dx^2 + dz^2 )
			distance = sqrt(dx*dx + dy*dy + dz*dz + softening);
			distance_cubed = distance * distance * distance;

			// Now compute G * m_2 * 1/distance^3 term, as we will be using this
			// term once for each dimension
			// NOTE: we do not include m_1 here, as when we compute the change in velocity
			// of particle 1 later, we would be dividing this out again, so just leave it out
			m_j = remote_bodies[j].mass;
			mGd = G * m_j / distance_cubed;

			Fx += mGd * dx;
			Fy += mGd * dy;
			Fz += mGd * dz;
		}

		// With the total forces on particle "i" known from this batch, we can then update its velocity
		// v = (F * t) / m_i
		// NOTE: as discussed above, we have left out m_1 from previous velocity computation,
		// so this is left out here as well

		local_bodies[i].vx += dt*Fx;
		local_bodies[i].vy += dt*Fy;
		local_bodies[i].vz += dt*Fz;
	}
}

void run_parallel_problem(int nBodies, double dt, int nIters, int nthreads, char * fname)
{
	// MPI Init
	int mype, nprocs;
	int DIMENSION = 1;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);


	if (mype == 0)
	{
		printf("INPUT PARAMETERS:\n");
		printf("N Bodies =                     %d\n", nBodies);
		printf("Timestep dt =                  %.3le\n", dt);
		printf("Number of Timesteps =          %d\n", nIters);
		printf("Number of Threads per Rank =   %d\n", nthreads);
		printf("Number of Ranks =   %d\n", nprocs);
		printf("BEGINNING N-BODY SIMLUATION\n");
	}

	// Open File
	MPI_File fh;
	MPI_File_open(MPI_COMM_WORLD, "nbody.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
	if (mype == 0)
	{
		MPI_File_write(fh, &nBodies, 1, MPI_INT,  MPI_STATUS_IGNORE);
		MPI_File_write(fh, &nIters, 1, MPI_INT,  MPI_STATUS_IGNORE);
	}
	
	// When we open the this binary file for plotting, we will make some assumptions as to
	// size of data types we are writing. As such, we enforce these assumptions here.
	assert(sizeof(int)    == 4 );
	assert(sizeof(double) == 8 );	

	// assert nprocs divides nbodies
	if ((nBodies%nprocs) != 0)
	{
		if (mype == 0)
			printf("nBodies needs to be divisible by nprocs\n");
		return;
	}

	//MPI Cartesian Grid Creation
	int periodic = 1; 
	MPI_Comm comm1d;

	// set up cartesian communicator
	MPI_Cart_create(MPI_COMM_WORLD,
		DIMENSION,
		&nprocs,
		&periodic,
		1,
		&comm1d
		);

	// prepare left/right communication
	int left_neighbor, right_neighbor;
	MPI_Cart_shift(comm1d,
		0,
		1,
		&left_neighbor,
		&right_neighbor);

	// create body mpi datatype
	MPI_Datatype body_mpi_type;
	MPI_Datatype type[7] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
	int 		 blocklen[7] = {1, 1, 1, 1, 1, 1, 1};
	MPI_Aint 	 disp[7] = {0, sizeof(double), 2*sizeof(double), 3*sizeof(double), 4*sizeof(double), 5*sizeof(double), 6*sizeof(double)};
	MPI_Type_create_struct(7, blocklen, disp, type, &body_mpi_type);
	MPI_Type_commit(&body_mpi_type);

	// initialize body arrays
	int bodies_per_rank = nBodies/nprocs;
	Body * local_bodies = (Body *) malloc(bodies_per_rank * sizeof(Body));
	Body * remote_bodies = (Body *) malloc(bodies_per_rank * sizeof(Body));
	Body * send_bodies = (Body *) malloc(bodies_per_rank * sizeof(Body));

	// Apply Randomized Initial Conditions to Bodies
	parallel_randomizeBodies(local_bodies, bodies_per_rank, nBodies, mype, nprocs);

	int nPositions = nBodies * 3;
	double * positions = (double *) malloc( nPositions * sizeof(double));

	// Start timer
	double start = 0;
	if (mype == 0)
		start = omp_get_wtime();

	// Loop over timesteps
	MPI_Offset total_offset;
	for (int iter = 0; iter < nIters; iter++)
	{
		if (mype == 0)
			printf("iteration: %d\n", iter);

		// Pack up body positions to contiguous buffer
		for( int b = 0, p = 0; b < bodies_per_rank; b++ )
		{
			positions[p++] = local_bodies[b].x;
			positions[p++] = local_bodies[b].y;
			positions[p++] = local_bodies[b].z;
		}

		total_offset = (sizeof(int) * 2) + (iter * sizeof(double) * nBodies * 3) + (mype * sizeof(double) * bodies_per_rank * 3);
		MPI_File_write_at_all(fh, total_offset, positions, bodies_per_rank * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
		//print_bodies(local_bodies, bodies_per_rank, mype, iter);

		for (int pipe_iter = 0; pipe_iter < nprocs; pipe_iter++)
		{

			if (pipe_iter == 0)
				compute_forces_multi_set(local_bodies, local_bodies, dt, bodies_per_rank, mype);
			else
				compute_forces_multi_set(local_bodies, remote_bodies, dt, bodies_per_rank, mype);

			// shift 
			if (pipe_iter < (nprocs-1))
			{
				if (pipe_iter == 0)
					memcpy(send_bodies, local_bodies, bodies_per_rank * sizeof(Body));
				else
					memcpy(send_bodies, remote_bodies, bodies_per_rank * sizeof(Body));

				MPI_Sendrecv(&send_bodies[0], bodies_per_rank, body_mpi_type, left_neighbor, 99,
							&remote_bodies[0], bodies_per_rank, body_mpi_type, right_neighbor,
							99, comm1d, MPI_STATUS_IGNORE);

			}

			MPI_Barrier(MPI_COMM_WORLD);

		}

		// Update positions of all particles
		for (int i = 0 ; i < bodies_per_rank; i++)
		{
			local_bodies[i].x += local_bodies[i].vx*dt;
			local_bodies[i].y += local_bodies[i].vy*dt;
			local_bodies[i].z += local_bodies[i].vz*dt;
		}

		MPI_Barrier(MPI_COMM_WORLD);

	}

	if (mype == 0)
	{
		// Stop timer and print stats
		double stop = omp_get_wtime();
		double runtime = stop-start;
		double time_per_iter = runtime / nIters;
		double interactions = (double) nBodies * (double) nBodies;
		double interactions_per_sec = interactions / time_per_iter;

		printf("SIMULATION COMPLETE\n");
		printf("Runtime [s]:              %.3le\n", runtime);
		printf("Runtime per Timestep [s]: %.3le\n", time_per_iter);
		printf("Interactions per sec:     %.3le\n", interactions_per_sec);
	}

	free(local_bodies);
	free(remote_bodies);
	free(positions);
}

int main(int argc, char* argv[])
{
	// Input Parameters
	long nBodies = 1000;
	double dt = 0.2; 
	int nIters = 200;
	int nthreads = 1;
	char * fname = "nbody.dat";

	if( argc != 5 )
	{
		printf("Usage: ./nbody_serial <number of bodies> <number of iterations> <timestep length (dt)> <number of OpenMP threads per rank>\n");
		printf("Using defaults for now...\n");
	}

	// Initialize MPI
	MPI_Init(&argc, &argv);

	if( argc > 1 )
		nBodies = atol(argv[1]);
	if( argc > 2 )
		nIters = atoi(argv[2]);
	if( argc > 3 )
		dt = atof(argv[3]);
	if( argc > 4 )
		nthreads = atoi(argv[4]);

	// Set number of OMP threads if necessary
	omp_set_num_threads(nthreads);

	// Run Problem
	run_parallel_problem(nBodies, dt, nIters, nthreads, fname);

	// Finalize MPI
	MPI_Finalize();

	return 0;
}

