#ifndef OPENMC_NEURAL_BOUNDARY_H
#define OPENMC_NEURAL_BOUNDARY_H

#include "openmc/particle.h"
#include "openmc/particle_data.h"

#include <map>
#include <vector>

#ifdef OPENMC_ONNX_ENABLED
#include <onnxruntime_cxx_api.h>

namespace openmc {

struct NeuralBCData {
  ParticleType particle;
  unsigned long surface_id;
  int cell_id;
  int64_t particle_id;
  Position centroid;
  Direction normal;
  Position r;
  Direction u;
  double E;
  double time {0.0};
  double wgt {1.0};
};

struct RunData {
  std::vector<int64_t> shape;
  std::vector<float> values;
  size_t size;
};

struct MapType {
  unsigned long entity;
  int64_t nnid;
};

// Ensure cache alignment
struct alignas(64) ONNXInput {
  // Outgoing facet
  std::vector<int64_t> s_shape;
  std::vector<int64_t> s_values;
  size_t s_size;
  // Outgoing continuous data
  std::vector<int64_t> c_shape;
  std::vector<float> c_values;
  size_t c_size;
};

//==============================================================================
// Non-member functions
//==============================================================================

extern std::vector<std::vector<NeuralBCData>> surrogate_boundary_crossings;
// Want to pre-allocate the vectors used to prepare tensor data
extern std::vector<ONNXInput> onnx_input_data;
extern std::vector<std::vector<Ort::Value>> onnx_input_tensors;

extern OrtEnv* onnx_environment;
extern Ort::Env* env;
extern std::vector<Ort::Session> onnx_model;
extern Ort::MemoryInfo onnx_memory_info;
extern Ort::RunOptions onnx_runoptions;
extern const char* onnx_inames[];
extern const char* onnx_onames[];
extern std::map<unsigned long, int64_t> onnx_map;
extern std::map<int64_t, unsigned long> onnx_map_inv;

// Allocate vector for number of threads
void initialize_train_surrogate_BC();

void initialize_train_surrogate_BC_batch();

// This should just append data to the array
void write_surrogate_BC_data(Particle& p, const Surface& surf);

// Write to hdf, clear vectors
void finalize_train_surrogate_BC_batch();

// Close the hdf file
void finalize_train_surrogate_BC();

void initialize_infer_surrogate_BC();

void infer_crossing_surrogate_BC(Particle& p, const Surface& surf);

void finalize_infer_surrogate_BC();

} // namespace openmc

#endif // OPENMC_ONNX_ENABLED

#endif // OPENMC_NEURAL_BOUNDARY_H
