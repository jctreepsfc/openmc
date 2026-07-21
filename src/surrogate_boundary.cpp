#include "openmc/surrogate_boundary.h"
#include "openmc/dagmc.h"
#include "openmc/geometry.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/position.h"
#include "openmc/simulation.h"
#include "openmc/surface.h"

#include <array>
#include <cmath> // For log
#include <fmt/core.h>
#include <hdf5.h>
#include <omp.h>

#ifdef OPENMC_ONNX_ENABLED
#include <onnxruntime_cxx_api.h>

namespace openmc {

hid_t boundary_file;
hid_t cross_dtype;

std::vector<std::vector<NeuralBCData>> surrogate_boundary_crossings;
std::vector<ONNXInput> onnx_input_data;
std::vector<std::vector<Ort::Value>> onnx_input_tensors;

OrtEnv* onnx_environment = nullptr;
Ort::Env* env = nullptr;
std::vector<Ort::Session> onnx_model;
Ort::MemoryInfo onnx_memory_info =
  Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::RunOptions onnx_runoptions {nullptr};
const char* onnx_inames[] = {
  "s_out", "c_out"}; //, "return_target", "tile_target",
                     //"angle_target", "energy_target"};
const char* onnx_onames[] = {"returns", "s_in", "angle", "energy"};
std::map<unsigned long, int64_t> onnx_map;
std::map<int64_t, unsigned long> onnx_map_inv;

void initialize_train_surrogate_BC()
{
  // Get the number of threads and pre-allocate
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    surrogate_boundary_crossings.push_back(std::vector<NeuralBCData> {});
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
    H5T_NATIVE_ULONG);
  H5Tinsert(
    cross_dtype, "cell_id", HOFFSET(NeuralBCData, cell_id), H5T_NATIVE_INT);
  H5Tinsert(cross_dtype, "particle_id", HOFFSET(NeuralBCData, particle_id),
    H5T_NATIVE_INT);
  H5Tinsert(cross_dtype, "centroid", HOFFSET(NeuralBCData, centroid), postype);
  H5Tinsert(cross_dtype, "normal", HOFFSET(NeuralBCData, normal), postype);
  H5Tinsert(cross_dtype, "r", HOFFSET(NeuralBCData, r), postype);
  H5Tinsert(cross_dtype, "u", HOFFSET(NeuralBCData, u), postype);
  H5Tinsert(cross_dtype, "E", HOFFSET(NeuralBCData, E), H5T_NATIVE_DOUBLE);
  H5Tinsert(
    cross_dtype, "time", HOFFSET(NeuralBCData, time), H5T_NATIVE_DOUBLE);
  H5Tinsert(cross_dtype, "wgt", HOFFSET(NeuralBCData, wgt), H5T_NATIVE_DOUBLE);

  H5Tclose(postype);
}

void initialize_train_surrogate_BC_batch()
{
  // Empty per-thread memory before a new batch
  // This is called outside of omp parallel
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    surrogate_boundary_crossings[i].clear();
  }
}

void write_surrogate_BC_data(Particle& p, const Surface& surf)
{
  // Get current thread's data vector
  std::vector<NeuralBCData>& bank_access =
    surrogate_boundary_crossings[omp_get_thread_num()];
  // Need last crossing to see if it is the same particle
  long last_particle = -1;
  if (!bank_access.empty())
    last_particle = bank_access.back().particle_id;

  // Get last facet crossing
  moab::EntityHandle facet;
  MB_CHK_ERR_CONT(p.history().get_last_intersection(facet));

  // Get facet normal
  Direction normal = surf.normal(p.r());
  if (p.u().dot(normal) < 0.)
    normal *= -1.;

  // Get facet centroid
  // NOTE: Put this here since I don't think the surface should logically have a
  // method for this
  auto dag_ptr = dynamic_cast<const DAGSurface&>(surf).dagmc_ptr();
  std::vector<moab::EntityHandle> vertices;
  dag_ptr->moab_instance()->get_adjacencies(&facet, 1, 0, false, vertices);
  std::vector<double> coords(9);
  dag_ptr->moab_instance()->get_coords(
    &vertices[0], vertices.size(), coords.data());
  Position centroid {(coords[0] + coords[3] + coords[6]) / 3.0,
    (coords[1] + coords[4] + coords[7]) / 3.0,
    (coords[2] + coords[5] + coords[8]) / 3.0};

  // Create the data
  NeuralBCData crossing;
  crossing.particle = p.type();
  crossing.surface_id = static_cast<unsigned long>(facet);
  crossing.cell_id = p.lowest_coord().cell();
  crossing.particle_id = p.id();
  crossing.centroid = centroid;
  crossing.normal = normal;
  crossing.r = p.r();
  crossing.u = p.u();
  crossing.E = log(p.E());
  crossing.time = p.time();
  crossing.wgt = p.wgt();
  bank_access.push_back(crossing);

  // Kill particle if it crosses back into component
  if (last_particle == p.id()) {
    p.wgt() = 0;
  }
}

void infer_crossing_surrogate_BC(Particle& p, const Surface& surf)
{
  std::vector<Ort::Value>& model_input =
    onnx_input_tensors[omp_get_thread_num()];
  ONNXInput& model_data = onnx_input_data[omp_get_thread_num()];

  // === CONSTRUCT TENSOR WITH FACET INFO ===

  unsigned long facet;
  MB_CHK_ERR_CONT(p.history().get_last_intersection(facet));
  // model_data.s_values = {static_cast<int64_t>(onnx_map.at(facet))};

  // === CONSTRUCT TENSOR WITH CONTINUOUS PARTICLE DATA ===

  auto r = p.r();
  auto u = p.u();
  auto E = log(p.E());
  // model_data.c_values = {static_cast<float>(r.x), static_cast<float>(r.y),
  //   static_cast<float>(r.z), static_cast<float>(u.x),
  //   static_cast<float>(u.y), static_cast<float>(u.z), static_cast<float>(E)};
  model_data.s_values[0] = static_cast<int64_t>(onnx_map.at(facet));
  model_data.c_values[0] = static_cast<float>(r.x);
  model_data.c_values[1] = static_cast<float>(r.y);
  model_data.c_values[2] = static_cast<float>(r.z);
  model_data.c_values[3] = static_cast<float>(u.x);
  model_data.c_values[4] = static_cast<float>(u.y);
  model_data.c_values[5] = static_cast<float>(u.z);
  model_data.c_values[6] = static_cast<float>(E);

  // === CONSTRUCT GUMBEL-MAX NOISE TENSORS ===

  // for (int i = 0; i < 4; ++i) {
  //   // Ex. for c_out: [-1, 7]
  //   // Create tensor
  //   for (int j = 0; j < model_data.n_sizes[i]; ++j) {
  //     model_data.n_values[i][j] = prn(p.current_seed());
  //   }
  // }

  // === RUN THE MODEL ===

  // NOTE: We do not create the tensors, since they have already been setup to
  // point at the memory of model.values
  auto out = onnx_model[omp_get_thread_num()].Run(
    onnx_runoptions, onnx_inames, model_input.data(), 2, onnx_onames, 4);

  // === MOVE THE PARTICLE ===

  p.r_last() = p.r();
  p.u_last() = p.u();
  p.E_last() = p.E();
  p.wgt_last() = p.wgt();

  float wgt_mult = out[0].GetTensorMutableData<float>()[0];
  if (wgt_mult < 1e-5) {
    p.wgt() = 0.;
    return;
  }
  p.wgt() *= wgt_mult;

  // Convert back from index to facet id
  int64_t facet_i = out[1].GetTensorMutableData<int64_t>()[0];
  facet = onnx_map_inv.at(facet_i);
  // Get the centroid of the facet
  auto dag_ptr = dynamic_cast<const DAGSurface&>(surf).dagmc_ptr();
  std::vector<moab::EntityHandle> vertex_handles;
  dag_ptr->moab_instance()->get_adjacencies(
    &facet, 1, 0, false, vertex_handles);
  std::vector<double> coords(9);
  dag_ptr->moab_instance()->get_coords(
    &vertex_handles[0], vertex_handles.size(), coords.data());

  p.r().x = (coords[0] + coords[3] + coords[6]) / 3.0;
  p.r().y = (coords[1] + coords[4] + coords[7]) / 3.0;
  p.r().z = (coords[2] + coords[5] + coords[8]) / 3.0;

  float* angle_i = out[2].GetTensorMutableData<float>();
  p.u().x = angle_i[0];
  p.u().y = angle_i[1];
  p.u().z = angle_i[2];

  p.E() = exp(out[3].GetTensorMutableData<float>()[0]);
  // Kill the particle if the energy goes to zero: avoids nan down the line
  if (p.E() == 0) {
    p.wgt() = 0;
    return;
  }

  p.history().reset();

  p.r_last_current() = p.r() + TINY_BIT * p.u();
  p.r() += TINY_BIT * p.u();
  p.surface() = SURFACE_NONE;
  // Figure out what cell particle is in now
  p.n_coord() = 1;
  if (!exhaustive_find_cell(p)) {
    p.mark_as_lost("Couldn't find particle after hitting surrogate "
                   "boundary on surface " +
                   std::to_string(surf.id_) + ".");
    return;
  }

  // Need to recalculate cross sections since our energy changed
  if (p.material() != MATERIAL_VOID)
    model::materials[p.material()]->calculate_xs(p);
}

void finalize_train_surrogate_BC_batch()
{
  for (int thread = 0; thread < surrogate_boundary_crossings.size(); thread++) {
    std::string dset_name =
      fmt::format("crossing_{}_{}", simulation::current_batch, thread);
    auto& bank_access = surrogate_boundary_crossings[thread];

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

void finalize_train_surrogate_BC()
{
  H5Tclose(cross_dtype);
  file_close(boundary_file);
}

void initialize_infer_surrogate_BC()
{
  OrtThreadingOptions* tp_options = nullptr;
  auto ret = Ort::GetApi().CreateThreadingOptions(&tp_options);
  // Set threads to 1 for intra-op to get 1 core per OpenMP thread
  ret = Ort::GetApi().SetGlobalIntraOpNumThreads(tp_options, 1);
  ret = Ort::GetApi().SetGlobalInterOpNumThreads(tp_options, 1);
  // Disable spinning to stop idle threads from hogging 100% CPU
  ret = Ort::GetApi().SetGlobalSpinControl(tp_options, 0);
  // Initialize the environment
  ret = Ort::GetApi().CreateEnvWithGlobalThreadPools(
    ORT_LOGGING_LEVEL_WARNING, "Model", tp_options, &onnx_environment);
  env = new Ort::Env(onnx_environment);
  // Clean up the helper (Env takes a copy)
  Ort::GetApi().ReleaseThreadingOptions(tp_options);

  // OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  // ret = Ort::GetApi().CreateCUDAProviderOptions(&cuda_options);
  // std::vector<const char*> keys {"device_id", "do_copy_in_default_stream"};
  // std::vector<const char*> values {"0", "0"};
  // ret = Ort::GetApi().UpdateCUDAProviderOptions(
  //   cuda_options, keys.data(), values.data(), (int)keys.size());

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.DisablePerSessionThreads();
  session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  // ret = Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
  //   session_options, cuda_options);
  // Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);

  std::string filename = fmt::format("{}model.onnx", settings::path_output);
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    onnx_model.emplace_back(*env, filename.c_str(), session_options);
  }

  // Read mapping
  hid_t maptype = H5Tcreate(H5T_COMPOUND, sizeof(struct MapType));
  H5Tinsert(maptype, "entity", HOFFSET(MapType, entity), H5T_NATIVE_ULONG);
  H5Tinsert(maptype, "nnid", HOFFSET(MapType, nnid), H5T_NATIVE_INT64);

  std::vector<MapType> map_data;
  // FIX: Need to make this take an arbitrary filename
  //      Also test that the file, dset, etc. exists
  hid_t mapfile = file_open("mappings.h5", 'r');
  hid_t dset = H5Dopen(mapfile, "inference_nnids", H5P_DEFAULT);
  hid_t dspace = H5Dget_space(dset);
  hsize_t n_sites;
  H5Sget_simple_extent_dims(dspace, &n_sites, nullptr);
  map_data.resize(n_sites);
  hid_t memspace = H5S_ALL;
  H5Dread(dset, maptype, memspace, dspace, H5P_DEFAULT, map_data.data());
  H5Sclose(dspace);
  H5Dclose(dset);
  H5Tclose(maptype);
  file_close(mapfile);

  for (MapType& entry : map_data) {
    onnx_map[entry.entity] = entry.nnid;
    onnx_map_inv[entry.nnid] = entry.entity;
  }

  // Pre-allocate memory for model inputs
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    ONNXInput input;
    input.s_shape = {1};
    input.s_values = std::vector<int64_t>(1);
    input.s_size = 1;
    input.c_shape = {1, 7};
    input.c_values = std::vector<float>(7);
    input.c_size = 7;
    // for (int j = 2; j < 6; ++j) {
    //   auto dims = onnx_model[i]
    //                 .GetInputTypeInfo(j)
    //                 .GetTensorTypeAndShapeInfo()
    //                 .GetShape()
    //                 .size();
    //   if (dims == 1) {
    //     input.n_shapes.push_back(std::vector<int64_t> {1});
    //     input.n_sizes.push_back(1);
    //     input.n_values.push_back(std::vector<float>(1));
    //   } else {
    //     int w = onnx_model[i]
    //               .GetInputTypeInfo(j)
    //               .GetTensorTypeAndShapeInfo()
    //               .GetShape()[1];
    //     input.n_shapes.push_back(std::vector<int64_t> {1, w});
    //     input.n_sizes.push_back(w);
    //     input.n_values.push_back(std::vector<float>(w));
    //   }
    // }
    onnx_input_data.push_back(input);
  }

  // Now we set up our tensors to point permanently at the data
  // NOTE: This allows us to never actually create a tensor during the loop
  onnx_input_tensors.resize(omp_get_max_threads());
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    auto& data = onnx_input_data[i];
    auto& input_tensors = onnx_input_tensors[i];
    input_tensors.reserve(6); // Total number of inputs
    // Bind s_values
    input_tensors.push_back(
      Ort::Value::CreateTensor<int64_t>(onnx_memory_info, data.s_values.data(),
        data.s_size, data.s_shape.data(), data.s_shape.size()));
    // Bind c_values
    input_tensors.push_back(
      Ort::Value::CreateTensor<float>(onnx_memory_info, data.c_values.data(),
        data.c_size, data.c_shape.data(), data.c_shape.size()));
    // Bind Noise data
    // for (size_t j = 0; j < data.n_values.size(); ++j) {
    //   input_tensors.push_back(Ort::Value::CreateTensor<float>(onnx_memory_info,
    //     data.n_values[j].data(), data.n_sizes[j], data.n_shapes[j].data(),
    //     data.n_shapes[j].size()));
    // }
  }
}

void finalize_infer_surrogate_BC()
{
  for (auto& tensors : onnx_input_tensors)
    tensors.clear();
  onnx_input_tensors.clear();

  onnx_model.clear();

  delete env;
  env = nullptr;
  onnx_environment = nullptr;

  onnx_input_data.clear();
  onnx_map.clear();
  onnx_map_inv.clear();
}

#endif // OPENMC_ONNX_ENABLED

} // namespace openmc
