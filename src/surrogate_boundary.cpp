#include "openmc/surrogate_boundary.h"
#include "openmc/bank.h"
#include "openmc/dagmc.h"
#include "openmc/geometry.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/memory.h"
#include "openmc/position.h"
#include "openmc/shared_array.h"
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

OrtEnv* onnx_environment = nullptr;
Ort::Env* env = nullptr;
Ort::Session onnx_model {nullptr};
Ort::MemoryInfo onnx_memory_info =
  Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::RunOptions onnx_runoptions {nullptr};
const char* onnx_inames[] = {"s_out", "c_out"};
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

void infer_crossing_surrogate_BC(
  std::vector<SurrogateSite>& bank, SharedArray<SourceSite>& shared_bank)
{
  int n_particles = bank.size();
  // Prepare the input data
  if (n_particles < 1) {
    return;
  }

  // int max_batch_size = 1 << 14;
  // int n_batches = n_particles / max_batch_size;
  int n_batches = 10;
  std::vector<int> batches;
  for (int i = 0; i < n_batches; ++i) {
    batches.push_back(n_particles / n_batches);
  }
  batches.back() += n_particles % n_batches;

  int working_offset = 0;
  for (int i = 0; i < n_batches; ++i) {
    int batch = batches[i];
    if (batch == 0)
      continue;

    ONNXInput input_data;
    input_data.s_shape = {batch};
    input_data.s_size = batch;
    input_data.c_shape = {batch, 7};
    input_data.c_size = batch * 7;
    input_data.s_values.reserve(batch);
    input_data.c_values.reserve(batch * 7);
    for (int j = working_offset; j < working_offset + batch; ++j) {
      auto& site = bank[j];
      input_data.s_values.push_back(onnx_map.at(site.facet));
      input_data.c_values.push_back(site.r.x);
      input_data.c_values.push_back(site.r.y);
      input_data.c_values.push_back(site.r.z);
      input_data.c_values.push_back(site.u.x);
      input_data.c_values.push_back(site.u.y);
      input_data.c_values.push_back(site.u.z);
      input_data.c_values.push_back(log(site.E));
    }

    // Create the tensors
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(onnx_memory_info,
      input_data.s_values.data(), input_data.s_size, input_data.s_shape.data(),
      input_data.s_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(onnx_memory_info,
      input_data.c_values.data(), input_data.c_size, input_data.c_shape.data(),
      input_data.c_shape.size()));

    auto out = onnx_model.Run(
      onnx_runoptions, onnx_inames, input_tensors.data(), 2, onnx_onames, 4);

    // Create the bank
    for (int j = 0; j < batch; ++j) {
      int bank_idx = working_offset + j;

      SourceSite tmp;
      float wgt_mult = out[0].GetTensorMutableData<float>()[j];
      if (wgt_mult < 1e-5)
        wgt_mult = 0.;
      tmp.wgt = bank[j].wgt * wgt_mult;
      int64_t facet_i = out[1].GetTensorMutableData<int64_t>()[j];
      unsigned long facet = onnx_map_inv.at(facet_i);

      auto& surf = model::surfaces[bank[j].surf_id];
      auto dag_ptr = dynamic_cast<const DAGSurface&>(*surf).dagmc_ptr();
      std::vector<moab::EntityHandle> vertex_handles;
      dag_ptr->moab_instance()->get_adjacencies(
        &facet, 1, 0, false, vertex_handles);
      std::vector<double> coords(9);
      dag_ptr->moab_instance()->get_coords(
        &vertex_handles[0], vertex_handles.size(), coords.data());
      tmp.r.x = (coords[0] + coords[3] + coords[6]) / 3.0;
      tmp.r.y = (coords[1] + coords[4] + coords[7]) / 3.0;
      tmp.r.z = (coords[2] + coords[5] + coords[8]) / 3.0;

      float* angle_i = out[2].GetTensorMutableData<float>();
      tmp.u.x = angle_i[j * 3];
      tmp.u.y = angle_i[j * 3 + 1];
      tmp.u.z = angle_i[j * 3 + 2];

      tmp.E = exp(out[3].GetTensorMutableData<float>()[j]);
      if (tmp.E == 0)
        tmp.wgt = 0.;
      // Advance off the surrogate surface by a little bit
      tmp.r += TINY_BIT * tmp.u;
      tmp.surf_id = SURFACE_NONE;
      // Add other info we need
      tmp.parent_id = bank[bank_idx].parent_id;
      tmp.progeny_id = bank[bank_idx].progeny_id;
      tmp.particle = bank[bank_idx].particle;
      tmp.wgt_born = bank[bank_idx].wgt_born;
      tmp.wgt_ww_born = bank[bank_idx].wgt_ww_born;
      tmp.n_split = bank[bank_idx].n_split;
      // Add this source site to the shared surrogate bank
      shared_bank.thread_unsafe_append(tmp);
    }
    working_offset += batch;
  }
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
  // Find what we are running on
  auto providers = Ort::GetAvailableProviders();
  bool has_gpu = false;
  bool has_rt = false;
  for (const auto& provider : providers) {
    if (provider == "CUDAExecutionProvider")
      has_gpu = true;
    if (provider == "TensorrtExecutionProvider")
      has_rt = true;
  }

  OrtThreadingOptions* tp_options = nullptr;
  auto ret = Ort::GetApi().CreateThreadingOptions(&tp_options);
  ret = Ort::GetApi().SetGlobalIntraOpNumThreads(tp_options, 1);
  ret = Ort::GetApi().SetGlobalInterOpNumThreads(tp_options, 1);
  ret = Ort::GetApi().SetGlobalSpinControl(tp_options, 0);
  ret = Ort::GetApi().CreateEnvWithGlobalThreadPools(
    ORT_LOGGING_LEVEL_WARNING, "Model", tp_options, &onnx_environment);
  env = new Ort::Env(onnx_environment);
  Ort::GetApi().ReleaseThreadingOptions(tp_options);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.DisablePerSessionThreads();
  session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  if (has_gpu) {
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    ret = Ort::GetApi().CreateCUDAProviderOptions(&cuda_options);
    std::vector<const char*> keys {"device_id", "do_copy_in_default_stream"};
    std::vector<const char*> values {"0", "1"};
    ret = Ort::GetApi().UpdateCUDAProviderOptions(
      cuda_options, keys.data(), values.data(), (int)keys.size());
    ret = Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
      session_options, cuda_options);
    Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);

    if (has_rt) {
      OrtTensorRTProviderOptionsV2* tensor_options = nullptr;
      ret = Ort::GetApi().CreateTensorRTProviderOptions(&tensor_options);
      ret = Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(
        session_options, tensor_options);
      Ort::GetApi().ReleaseTensorRTProviderOptions(tensor_options);
    }
  }

  std::string filename = fmt::format("{}model.onnx", settings::path_output);
  onnx_model = Ort::Session(*env, filename.c_str(), session_options);

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

  int max = 0;
  for (MapType& entry : map_data) {
    if (entry.nnid > max)
      max = entry.nnid;
    onnx_map[entry.entity] = entry.nnid;
    onnx_map_inv[entry.nnid] = entry.entity;
  }
  write_message(3, "Max nnid: {}", max);
}

void finalize_infer_surrogate_BC()
{
  onnx_model.release();

  delete env;
  env = nullptr;
  onnx_environment = nullptr;

  onnx_map.clear();
  onnx_map_inv.clear();
}

#endif // OPENMC_ONNX_ENABLED

} // namespace openmc
