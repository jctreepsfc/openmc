#ifndef OPENMC_NEURAL_BOUNDARY_H
#define OPENMC_NEURAL_BOUNDARY_H

#include "openmc/particle.h"
#include "openmc/particle_data.h"

#include <vector>

namespace openmc {

struct NeuralBCData {
  ParticleType particle;
  unsigned long surface_id;
  int64_t particle_id;
  Position centroid;
  Direction normal;
  Position r;
  Direction u;
  double E;
  double time {0.0};
  double wgt {1.0};
};

//==============================================================================
// Non-member functions
//==============================================================================

extern std::vector<std::vector<NeuralBCData>> neural_boundary_crossings;

// Allocate vector for number of threads
void initialize_neural_BC();

void initialize_neural_BC_batch();

// This should just append data to the array
void write_neural_BC_data(Particle& p, const Surface& surf);

// Write to hdf, clear vectors
void finalize_neural_BC_batch();

// Close the hdf file
void finalize_neural_BC();

} // namespace openmc

#endif // OPENMC_NEURAL_BOUNDARY_H
