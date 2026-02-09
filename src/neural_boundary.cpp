#include "openmc/neural_boundary.h"
#include "openmc/hdf5_interface.h"
#include "openmc/position.h"
#include "openmc/simulation.h"
#include "openmc/surface.h"

#include <array>
#include <fmt/core.h>
#include <hdf5.h>
#include <omp.h>

namespace openmc {

hid_t boundary_file;
hid_t cross_dtype;

std::vector<std::vector<NeuralBCData>> neural_boundary_crossings;

void initialize_neural_BC()
{
  // Get the number of threads and pre-allocate
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    neural_boundary_crossings.push_back(std::vector<NeuralBCData> {});
  }

  // Open hdf5 file
  std::string filename = fmt::format("{}boundary.h5", settings::path_output);
  boundary_file = file_open(filename, 'w');
  write_attribute(boundary_file, "filetype", "boundary");
  // TODO: Replace with a VERSION from constants.h
  write_attribute(boundary_file, "version", std::array<int, 2> {3, 0});

  // Compound type for position
  hid_t postype = H5Tcreate(H5T_COMPOUND, sizeof(struct Position));
  H5Tinsert(postype, "x", HOFFSET(Position, x), H5T_NATIVE_DOUBLE);
  H5Tinsert(postype, "y", HOFFSET(Position, y), H5T_NATIVE_DOUBLE);
  H5Tinsert(postype, "z", HOFFSET(Position, z), H5T_NATIVE_DOUBLE);

  // Compount type for NeuralBCData
  cross_dtype = H5Tcreate(H5T_COMPOUND, sizeof(struct NeuralBCData));
  H5Tinsert(
    cross_dtype, "particle", HOFFSET(NeuralBCData, particle), H5T_NATIVE_INT);
  H5Tinsert(cross_dtype, "surface_id", HOFFSET(NeuralBCData, surface_id),
    H5T_NATIVE_INT);
  H5Tinsert(cross_dtype, "particle_id", HOFFSET(NeuralBCData, particle_id),
    H5T_NATIVE_INT);
  H5Tinsert(cross_dtype, "r", HOFFSET(NeuralBCData, r), postype);
  H5Tinsert(cross_dtype, "u", HOFFSET(NeuralBCData, u), postype);
  H5Tinsert(cross_dtype, "E", HOFFSET(NeuralBCData, E), H5T_NATIVE_DOUBLE);
  H5Tinsert(
    cross_dtype, "time", HOFFSET(NeuralBCData, time), H5T_NATIVE_DOUBLE);
  H5Tinsert(cross_dtype, "wgt", HOFFSET(NeuralBCData, wgt), H5T_NATIVE_DOUBLE);

  H5Tclose(postype);
}

void initialize_neural_BC_batch()
{
  // Not sure if this is necessary yet
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    neural_boundary_crossings[i].clear();
  }
}

void write_neural_BC_data(Particle& p, const Surface& surf)
{
  // Get current thread's data vector
  std::vector<NeuralBCData>& bank_access =
    neural_boundary_crossings[omp_get_thread_num()];
  // Get last facet crossing
  // TODO: Will this be the current boundary crossing
  unsigned long facet;
  MB_CHK_ERR_CONT(p.history().get_last_intersection(facet));
  // Create the data
  NeuralBCData crossing;
  crossing.particle = p.type();
  crossing.surface_id = facet; // surf.id_;
  crossing.particle_id = p.id();
  crossing.r = p.r();
  crossing.u = p.u();
  crossing.E = p.E();
  crossing.time = p.time();
  crossing.wgt = p.wgt();
  bank_access.push_back(crossing);
}

void finalize_neural_BC_batch()
{
  for (int thread = 0; thread < neural_boundary_crossings.size(); thread++) {
    std::string dset_name =
      fmt::format("crossing_{}_{}", simulation::current_batch, thread);
    auto& bank_access = neural_boundary_crossings[thread];

    hsize_t dims[] {static_cast<hsize_t>(bank_access.size())};
    hid_t dspace = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate(boundary_file, dset_name.c_str(), cross_dtype,
      dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(
      dset, cross_dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, bank_access.data());
    H5Dclose(dset);
    H5Sclose(dspace);
  }
}

void finalize_neural_BC()
{
  H5Tclose(cross_dtype);
  file_close(boundary_file);
}

} // namespace openmc
