// Copyright 2026 Pierre Talbot, Frederic Pinel

#include <limits>
#include <random>
#include <iostream>
#include "battery/vector.hpp"
#include "battery/allocator.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/memory.hpp"
#include "lala/ub.hpp"
#include "lala/fixpoint.hpp"

namespace cg = ::cooperative_groups;
namespace bt = ::battery;
using namespace lala;

void init_random_vector(auto& v, size_t to) {
  std::mt19937 m{std::random_device{}()};
  std::uniform_int_distribution<size_t> dist(0, to);
  for (size_t i=0; i < v.size(); i+=2) {
      v[i] = dist(m);
      do { v[i+1] = dist(m); } 
      while (v[i] == v[i+1]);
      // duplicate edges are acceptable
  }
}

void init_distances(auto& v) {
    // v[0] is the sink (or target)
    v[0].meet_bot();
    for (size_t i=1; i < v.size(); ++i) {
        v[i].join_top();
    }
}

struct MinimumData {
  int iterations;
  bt::vector<size_t, bt::managed_allocator> data;  // each edge defined by pair of vertices: V2 V4 ...
  bt::vector<UB<size_t>, bt::managed_allocator> min_val;  // minimum distances from i to sink

  CUDA MinimumData(size_t edges, size_t vertices)
  : data(edges*2), min_val(vertices)
  {}
};

__global__ void compute_min(MinimumData* m_ptr) {
  MinimumData& m = *m_ptr;
  assert(gridDim.x != 0);
  __shared__ BlockAsynchronousFixpointGPU<true> fp_engine;
  size_t block_size = m.data.size() / gridDim.x;
  __syncthreads();
  m.iterations = fp_engine.fixpoint(
    block_size,
    [&](int i) { return m.min_val.meet(m.data[block_size * blockIdx.x + i]); }
  );
}

__global__ void compute_min_grid(MinimumData* m_ptr) {
  MinimumData& m = *m_ptr;
  bt::unique_ptr<GridAsynchronousFixpointGPU, bt::global_allocator> fp_engine_ptr;
  auto g = cg::this_grid();
  GridAsynchronousFixpointGPU& fp_engine = bt::make_unique_grid(fp_engine_ptr, g);
  m.iterations = fp_engine.fixpoint(
    m.data.size(),
    [&](int i) { return m.min_val.meet(m.data[i]); }
  );
  g.sync();
}

#define TPB 256
#define VERTICES 6
#define EDGES 2<<3

int main(int argc, char** argv) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  auto gpu_data = bt::make_unique<MinimumData, bt::managed_allocator>(EDGES, VERTICES);
  init_random_vector(gpu_data->data, VERTICES -1);
  init_distances(gpu_data->min_val);

  std::cout << "Data initialized. Starting GPU benchmarks." << std::endl;

  // On CPU, using a sequential fixpoint algorithm.
  GaussSeidelIteration fp_engine;
  fp_engine.iterate(gpu_data->data.size(), [&](int i) { return gpu_data->min_val.meet(gpu_data->data[i]); });
  std::cout << "CPU Minimum: " << std::endl;
  std::cout << "  minimum: " << cpu_min_val << std::endl;

  // On GPU, using a block-parallel fixpoint algorithm.
  gpu_data->min_val.join_top();
  int blocks = 1;
  std::cout << "GPU Minimum with " << blocks <<  " block(s)" << std::endl;
  std::cout << "  initially: " << gpu_data->min_val << std::endl;
  cudaEventRecord(start, 0);
  compute_min<<<blocks, TPB>>>(gpu_data.get());
  CUDAEX(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "  minimum: " << gpu_data->min_val;
  std::cout << "  iterations: " << gpu_data->iterations;
  std::cout << "  elapsed: " << elapsedTime << " ms" << std::endl;
  if(cpu_min_val != gpu_data->min_val) {
    std::cout << "!! error detected" << std::endl;
  }

  // On GPU, using a grid-parallel fixpoint algorithm.
  gpu_data->min_val.join_top();
  int device = 0, maxShmemSz = 0, nProc = 0, maxBlocksPerSM = 0;
  cudaGetDevice(&device);
  blocks = 2;
  cudaDeviceGetAttribute(&maxShmemSz, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cudaDeviceGetAttribute(&nProc, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, compute_min_grid, TPB, 0);
  blocks = std::min(blocks, maxBlocksPerSM * nProc);
  std::cout << "GPU Minimum on " << blocks <<  " blocks (max: " << maxBlocksPerSM  << " * " << nProc << " = " << maxBlocksPerSM * nProc << ")";
  std::cout << " with cooperative_groups";
  std::cout << " shared mem max: " << maxShmemSz/1024 << " kiB" << std::endl;
  std::cout << "  initially: " << gpu_data->min_val << std::endl;
  dim3 dimBlock(TPB, 1, 1);
  dim3 dimGrid(blocks, 1, 1);  
  auto pdata = gpu_data.get();
  void *args[] = { &pdata };
  cudaEventRecord(start, 0);
  CUDAEX( cudaLaunchCooperativeKernel(compute_min_grid, dimGrid, dimBlock, args) );
  CUDAEX( cudaDeviceSynchronize() );
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "  minimum: " << gpu_data->min_val;
  std::cout << "  iterations: " << gpu_data->iterations;
  std::cout << "  elapsed: " << elapsedTime << " ms" << std::endl;
  if(cpu_min_val != gpu_data->min_val) {
    std::cout << "!! error detected" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
