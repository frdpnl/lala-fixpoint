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

void init_random_edges(auto& e, unsigned int to) {
  std::mt19937 m{std::random_device{}()};
  // 0 is index of sink:
  std::uniform_int_distribution<unsigned int> rand_vertice(0, to);
  for (size_t i=0; i < e.size(); i+=2) {
      e[i] = rand_vertice(m);
      do { e[i+1] = rand_vertice(m); } 
      while (e[i] == e[i+1]);
      if (e[i] > e[i+1]) {
          std::swap(e[i], e[i+1]);
      }
      // duplicate edges are acceptable. just almost wasted computation
      // almost, because fixpoint helps in this case.
      // to expose TODO
  }
}

void init_distances(auto& d) {
    // d[0] is the sink (or target)
    d[0].meet_bot();
    for (size_t i=1; i < d.size(); ++i) {
        d[i].join_top();
    }
}

struct MinimumData {
  int iterations;
  bt::vector<unsigned int, bt::managed_allocator> edge;  // each edge defined by pair of vertices: V2 V4 ...
  bt::vector<UB<unsigned int>, bt::managed_allocator> min_dist;  // minimum distances from i to sink

  CUDA MinimumData(unsigned int vertices, unsigned int edges)
  : edge(edges*2), min_dist(vertices)
  {}
};

/*
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
*/

/*
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
*/

#define TPB 256
#define VERTICES 8
#define EDGES (VERTICES * 2)

int main(int argc, char** argv) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime = 0;

  auto gpu_data = bt::make_unique<MinimumData, bt::managed_allocator>(VERTICES, EDGES);
  init_random_edges(gpu_data->edge, VERTICES -1);
  init_distances(gpu_data->min_dist);

  std::cout << "Edges " << EDGES << ", distances " << VERTICES << " initialized. Starting benchmarks." << std::endl;
  std::cout << "\tedges: ";
  // poor man's sort:
  for (size_t v = 0; v < VERTICES; ++v) {
      std::cout << v << "->";
      for (size_t left = 0; left < gpu_data->edge.size()/2; ++left) {
          if (gpu_data->edge[2*left] == v) {
            std::cout << gpu_data->edge[2*left+1];
            std::cout << ",";
          }
      }
      std::cout << " ";
  }
  std::cout << std::endl;

  std::cout << "\tinitial distances: ";
  for (size_t i = 0; i < gpu_data->min_dist.size(); ++i) {
        std::cout << "d[" << i << "]=" << gpu_data->min_dist[i] << " ";
  }
  std::cout << std::endl;

  // On CPU, using a sequential fixpoint algorithm.
  GaussSeidelIteration fp_engine;
  fp_engine.fixpoint(
          gpu_data->edge.size()/2,  /* /2: edges are pairs of vertices */
          [&](int i) {  unsigned int left_ver = gpu_data->edge[2*i];
                        unsigned int right_ver = gpu_data->edge[2*i+1];
                        if (gpu_data->min_dist[left_ver].is_top()) {
                            return false;
                        }
                        return gpu_data->min_dist[right_ver].meet(1 + gpu_data->min_dist[left_ver]);
                     }
  );

  std::cout << "\tminimum distances: ";
  for (size_t i = 0; i < gpu_data->min_dist.size(); ++i) {
        std::cout << "d[" << i << "]=" << gpu_data->min_dist[i] << " ";
  }
  std::cout << std::endl;

  /*
  // On GPU, using a block-parallel fixpoint algorithm.
  init_distances(gpu_data->min_dist);
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
  init_distances(gpu_data->min_dist);
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
  */

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
