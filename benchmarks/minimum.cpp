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

namespace cg = cooperative_groups;

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
  assert(gridDim.x != 0);
  __shared__ BlockAsynchronousFixpointGPU<true> fp_engine;
  size_t block_size = m.data.size() / gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
      printf("--- kernel: tid=%3d, block_size=%3lu\n", tid, block_size);
  }
  __syncthreads();
  m.iterations = fp_engine.fixpoint(
    block_size,
    [&](int i) { return m.min_val.meet(m.data[block_size * blockIdx.x + i]); }
  );
}

__global__ void compute_min_g(MinimumData* m_ptr) {
  MinimumData& m = *m_ptr;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t block_size = m.data.size() / gridDim.x;
  if (threadIdx.x == 0) {
      printf("--- kernel: tid=%3d, block_size=%3lu\n", tid, block_size);
  }
  __shared__ AsynchronousIterationGPU fp_engine;
  m.iterations = fp_engine.fixpoint(
    block_size,
    [&](int i) { return m.min_val.meet(m.data[block_size * blockIdx.x + i]); }
  );
}

#define TPB 256
#define NUM_SECTIONS (2<<16)

int main(int argc, char** argv) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  size_t n = TPB * NUM_SECTIONS;
  auto gpu_data = bt::make_unique<MinimumData, bt::managed_allocator>(n);
  init_random_vector(gpu_data->data);

  std::cout << "Data initialized. Starting GPU benchmarks." << std::endl;

  // On CPU, using a sequential fixpoint algorithm.
  GaussSeidelIteration fp_engine;
  UB<int> cpu_min_val;
  fp_engine.iterate(gpu_data->data.size(), [&](int i) { return cpu_min_val.meet(gpu_data->data[i]); });
  std::cout << "CPU Minimum: " << std::endl;
  std::cout << "  minimum: " << cpu_min_val << std::endl;

  // On GPU, using a block-parallel fixpoint algorithm.
  gpu_data->min_val.join_top();
  std::cout << "GPU Minimum 1 block" << std::endl;
  std::cout << "  initially: " << gpu_data->min_val << std::endl;
  cudaEventRecord(start, 0);
  compute_min<<<1, TPB>>>(gpu_data.get());
  CUDAEX(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "  minimum: " << gpu_data->min_val;
  std::cout << "  iterations: " << gpu_data->iterations;
  std::cout << "  elapsed (ms): " << elapsedTime << std::endl;
  if(cpu_min_val != gpu_data->min_val) {
    std::cout << "!! error detected" << std::endl;
  }

  // On GPU, using a block-parallel fixpoint algorithm.
  gpu_data->min_val.join_top();
  std::cout << "GPU Minimum 2 blocks, no synchronization (benchmark only)" << std::endl;
  std::cout << "  initially: " << gpu_data->min_val << std::endl;
  cudaEventRecord(start, 0);
  compute_min<<<2, TPB>>>(gpu_data.get());
  CUDAEX(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "  minimum: " << gpu_data->min_val;
  std::cout << "  iterations: " << gpu_data->iterations;
  std::cout << "  elapsed (ms): " << elapsedTime << std::endl;
  if(cpu_min_val != gpu_data->min_val) {
    std::cout << "!! error detected" << std::endl;
  }

  // On GPU, using a grid parallel fixpoint algorithm.
  int device = 0, maxShmemSz = 0;
  cudaGetDevice(&device);
  gpu_data->min_val.join_top();
  std::cout << "GPU Minimum 2 blocks with cooperative_groups" << std::endl;
  std::cout << "  initially: " << gpu_data->min_val << std::endl;
  cudaDeviceGetAttribute(&maxShmemSz, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  //cudaDeviceProp deviceProp;
  //cudaGetDeviceProperties(&deviceProp, device);
  int maxBlocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, compute_min_g, TPB, 0);
  dim3 dimBlock(TPB, 1, 1);
  dim3 dimGrid(2, 1, 1);  
  //dim3 dimGrid(deviceProp.multiProcessorCount*maxBlocks, 1, 1);
  //size_t shmemSz = sizeof(AsynchronousIterationGPU);
  std::cout << "  maxBlocks: " << maxBlocks;
  std::cout << "  shared mem: max=" << maxShmemSz << std::endl;
  auto pdata = gpu_data.get();
  void *args[] = { &pdata };
  cudaEventRecord(start, 0);
  CUDAEX( cudaLaunchCooperativeKernel(compute_min_g, dimGrid, dimBlock, args) );
  CUDAEX( cudaDeviceSynchronize() );
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "  minimum: " << gpu_data->min_val;
  std::cout << "  iterations: " << gpu_data->iterations;
  std::cout << "  elapsed (ms): " << elapsedTime << std::endl;
  if(cpu_min_val != gpu_data->min_val) {
    std::cout << "!! error detected" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
