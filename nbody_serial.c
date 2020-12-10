#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>

// Code is modified version of https://github.com/jtramm/nbody_serial

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

void print_inputs(long nBodies, double dt, int nIters, int nthreads )
{

	printf("INPUT PARAMETERS:\n");
	printf("N Bodies =                     %ld\n", nBodies);
	printf("Timestep dt =                  %.3le\n", dt);
	printf("Number of Timesteps =          %d\n", nIters);
	printf("Number of Threads per Rank =   %d\n", nthreads);
	printf("BEGINNING N-BODY SIMLUATION\n");

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
void randomizeBodies(Body * bodies, int nBodies)
{

	uint64_t seed = 42;

	// velocity scaling term
	double vm = 1.0e-2;

	double rect_val;
	double x_rect_offset, y_rect_offset, z_rect_offset;
	double x_vel_direction, y_vel_direction;

	for (int i = 0; i < nBodies; i++)
	{
		// Fast forward seed to this particle's location in the global PRNG stream.
		// We forward 8 x particle_id, as each particle requires 8 PRNG samples.
		uint64_t particle_seed = fast_forward_LCG(seed, 8*i);

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

// Computes the forces between all bodies and updates
// their velocities accordingly
void compute_forces(Body * bodies, double dt, int nBodies)
{
	double G = 6.67259e-3;
	double softening = 1.0e-5;

	// For each particle in the set
	for (int i = 0; i < nBodies; i++)
	{ 
		double Fx = 0.0;
		double Fy = 0.0;
		double Fz = 0.0;

		// Compute force from all other particles in the set
		for (int j = 0; j < nBodies; j++)
		{
			// F_ij = G * [ (m_i * m_j) / distance^3 ] * (location_j - location_i) 

			// First, compute the "location_j - location_i" values for each dimension
			double dx = bodies[j].x - bodies[i].x;
			double dy = bodies[j].y - bodies[i].y;
			double dz = bodies[j].z - bodies[i].z;

			// Then, compute the distance^3 value
			// We will also include a "softening" term to prevent near infinite forces
			// for particles that come very close to each other (helps with stability)

			// distance = sqrt( dx^2 + dx^2 + dz^2 )
			double distance = sqrt(dx*dx + dy*dy + dz*dz + softening);
			double distance_cubed = distance * distance * distance;

			// Now compute G * m_2 * 1/distance^3 term, as we will be using this
			// term once for each dimension
			// NOTE: we do not include m_1 here, as when we compute the change in velocity
			// of particle 1 later, we would be dividing this out again, so just leave it out
			double m_j = bodies[j].mass;
			double mGd = G * m_j / distance_cubed;
			
			Fx += mGd * dx;
			Fy += mGd * dy;
			Fz += mGd * dz;
		}

		// With the total forces on particle "i" known from this batch, we can then update its velocity
		// v = (F * t) / m_i
		// NOTE: as discussed above, we have left out m_1 from previous velocity computation,
		// so this is left out here as well
		bodies[i].vx += dt*Fx;
		bodies[i].vy += dt*Fy;
		bodies[i].vz += dt*Fz;
	}
}

void print_bodies(Body * bodies, int nBodies, int iteration)
{
	// for debugging
	for (int i=0; i<nBodies; i++)
		printf("iteration %d, body %d, (x, y, z): %f, %f, %f, (vx, vy, vz): %f, %f, %f mass: %f\n", iteration, i, bodies[i].x, bodies[i].y, bodies[i].z, bodies[i].vx, bodies[i].vy, bodies[i].vz, bodies[i].mass);

}

void run_serial_problem(int nBodies, double dt, int nIters, char * fname)
{
	// Open File
	FILE * datafile = fopen("nbody.dat","wb");
	
	// When we open the this binary file for plotting, we will make some assumptions as to
	// size of data types we are writing. As such, we enforce these assumptions here.
	assert(sizeof(int)    == 4 );
	assert(sizeof(double) == 8 );

	// Write Header Info
	fwrite(&nBodies, sizeof(int), 1, datafile);
	fwrite(&nIters,  sizeof(int), 1, datafile);

	// Allocate Bodies
	Body * bodies  = (Body *) calloc( nBodies, sizeof(Body) );
	assert(bodies != NULL);

	// Apply Randomized Initial Conditions to Bodies
	randomizeBodies(bodies, nBodies);

	// Allocate additional space for contiguosly stored Cartesian body positions for easier file I/O
	int nPositions = nBodies * 3;
	double * positions = (double *) malloc( nPositions * sizeof(double));
	assert(positions != NULL);

	// Start timer
	double start = omp_get_wtime();

	// Loop over timesteps
	for (int iter = 0; iter < nIters; iter++)
	{
		printf("iteration: %d\n", iter);

		// Pack up body positions to contiguous buffer
		for( int b = 0, p = 0; b < nBodies; b++ )
		{
			positions[p++] = bodies[b].x;
			positions[p++] = bodies[b].y;
			positions[p++] = bodies[b].z;
		}

		// Output contiguous body positions to file
		fwrite(positions, sizeof(double), nPositions, datafile);
		//print_bodies(bodies, nBodies, iter);

		// Compute new forces & velocities for all particles
		compute_forces(bodies, dt, nBodies);

		// Update positions of all particles
		for (int i = 0 ; i < nBodies; i++)
		{
			bodies[i].x += bodies[i].vx*dt;
			bodies[i].y += bodies[i].vy*dt;
			bodies[i].z += bodies[i].vz*dt;
		}

	}

	// Close data file
	fclose(datafile);

	// Stop timer
	double stop = omp_get_wtime();

	double runtime = stop-start;
	double time_per_iter = runtime / nIters;
	double interactions = (double) nBodies * (double) nBodies;
	double interactions_per_sec = interactions / time_per_iter;

	printf("SIMULATION COMPLETE\n");
	printf("Runtime [s]:              %.3le\n", runtime);
	printf("Runtime per Timestep [s]: %.3le\n", time_per_iter);
	printf("Interactions per sec:     %.3le\n", interactions_per_sec);

	free(bodies);
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

	print_inputs(nBodies, dt, nIters, nthreads);
	run_serial_problem(nBodies, dt, nIters, fname);

	return 0;
}
