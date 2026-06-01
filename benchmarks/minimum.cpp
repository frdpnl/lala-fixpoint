// Copyright 2026 Pierre Talbot

#include <limits>
#include <random>
#include <iostream>
#include "battery/vector.hpp"
#include "battery/allocator.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/memory.hpp"
#include "lala/ub.hpp"
#include "lala/fixpoint.hpp"

namespace bt = ::battery;
using namespace lala;

void init_random_vector(auto& v) {
  std::mt19937 m{std::random_device{}()};
  std::uniform_int_distribution<int> dist{-1000000000, 1000000000};
  for(size_t i = 0; i < v.size(); ++i) {
    v[i] = dist(m);
  }
}

struct MinimumData {
  // using mem_type = bt::atomic_memory_grid;
  // using mem_type = bt::atomic_memory_scoped<cuda::thread_scope_device, cuda::memory_order_seq_cst>;
  // UB<int, mem_type> min_val;
  bt::vector<int, bt::managed_allocator> data;
  UB<int> min_val;
  int iterations;
  CUDA MinimumData(size_t n)
  : data(n)
  {}
};

__global__ void compute_min(MinimumData* m_ptr) {
  MinimumData& m = *m_ptr;
  assert(m.data.size() % (blockDim.x * gridDim.x) == 0);
  assert(gridDim.x == 0);
  __shared__ BlockAsynchronousFixpointGPU<true> fp_engine;
  size_t chunk_size = m.data.size() / gridDim.x;
  __syncthreads();
  m.iterations = fp_engine.fixpoint(
    chunk_size,
    [&](int i) { return m.min_val.meet(m.data[chunk_size * blockIdx.x + i]); }
  );
}

#define TPB 256
#define NUM_BLOCKS 1
#define NUM_SECTIONS 1000000

int main(int argc, char** argv) {
  size_t n = TPB * NUM_BLOCKS * NUM_SECTIONS;
  auto gpu_data = bt::make_unique<MinimumData, bt::managed_allocator>(n);
  init_random_vector(gpu_data->data);

  std::cout << "Data initialized. Starting GPU benchmarks." << std::endl;

  // On GPU, using a block-parallel fixpoint algorithm.
  compute_min<<<NUM_BLOCKS, TPB>>>(gpu_data.get());
  CUDAEX(cudaDeviceSynchronize());
  std::cout << "GPU Minimum: " << gpu_data->min_val << std::endl;
  std::cout << "  iterations: " << gpu_data->iterations << std::endl;

  // On CPU, using a sequential fixpoint algorithm.
  GaussSeidelIteration fp_engine;
  UB<int> cpu_min_val;
  fp_engine.iterate(gpu_data->data.size(), [&](int i) { return cpu_min_val.meet(gpu_data->data[i]); });
  std::cout << "CPU Minimum: " << cpu_min_val << std::endl;
  if(cpu_min_val != gpu_data->min_val) {
    std::cout << "!! error detected" << std::endl;
  }
}
