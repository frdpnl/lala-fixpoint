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

__global__ void compute_min_grid(MinimumData* m_ptr) {
  MinimumData& m = *m_ptr;
  bt::unique_ptr<GridAsynchronousFixpointGPU, bt::global_allocator> fp_engine_ptr;
  auto g = cg::this_grid();
  GridAsynchronousFixpointGPU& fp_engine = bt::make_unique_grid(fp_engine_ptr, g);
  m.iterations = fp_engine.fixpoint(
      m.edge.size()/2,  /* /2: edges are pairs of vertices */
      [&](int i) {  unsigned int left_ver = m.edge[2*i];
                    unsigned int right_ver = m.edge[2*i+1];
                    bool changed = false;
                    if (/* right_ver>0 && */ !m.min_dist[left_ver].is_top()) {
                        changed |= m.min_dist[right_ver].meet(1 + m.min_dist[left_ver]);
                    }
                    if (left_ver>0 && !m.min_dist[right_ver].is_top()) {
                        changed |= m.min_dist[left_ver].meet(1 + m.min_dist[right_ver]);
                    }
                    return changed;
                 }
  );
  g.sync();
}

#define TPB 256
#define VERTICES 4 
#define EDGES (VERTICES * 2)

int main(int argc, char** argv) {

  auto cpu_data = bt::make_unique<MinimumData, bt::managed_allocator>(VERTICES, EDGES);
  init_random_edges(cpu_data->edge, VERTICES -1);
  init_distances(cpu_data->min_dist);

  std::cout << "Edges " << EDGES << ", distances " << VERTICES << " initialized. Starting benchmarks." << std::endl;
  std::cout << "CPU Minimum" << std::endl;
  std::cout << "\tedges: ";
  // poor man's sort:
  for (size_t v = 0; v < VERTICES; ++v) {
      std::cout << v << "->";
      for (size_t left = 0; left < cpu_data->edge.size()/2; ++left) {
          if (cpu_data->edge[2*left] == v) {
            std::cout << cpu_data->edge[2*left+1];
            std::cout << ",";
          }
      }
      std::cout << " ";
  }
  std::cout << std::endl;

  std::cout << "\tinitial distances: ";
  for (size_t i = 0; i < cpu_data->min_dist.size(); ++i) {
        std::cout << "d[" << i << "]=" << cpu_data->min_dist[i] << " ";
  }
  std::cout << std::endl;

  // On CPU, using a sequential fixpoint algorithm.
  GaussSeidelIteration fp_engine;
  cpu_data->iterations = fp_engine.fixpoint(
          cpu_data->edge.size()/2,  /* /2: edges are pairs of vertices */
          [&](int i) {  unsigned int left_ver = cpu_data->edge[2*i];
                        unsigned int right_ver = cpu_data->edge[2*i+1];
                        bool changed = false;
                        if (/* right_ver>0 && */ !cpu_data->min_dist[left_ver].is_top()) {
                            changed |= cpu_data->min_dist[right_ver].meet(1 + cpu_data->min_dist[left_ver]);
                        }
                        if (left_ver>0 && !cpu_data->min_dist[right_ver].is_top()) {
                            changed |= cpu_data->min_dist[left_ver].meet(1 + cpu_data->min_dist[right_ver]);
                        }
                        return changed;
                     }
  );

  std::cout << "\tminimum distances: ";
  for (size_t i = 0; i < cpu_data->min_dist.size(); ++i) {
        std::cout << "d[" << i << "]=" << cpu_data->min_dist[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "\titerations: " << cpu_data->iterations << std::endl;

  // On GPU, using a grid-parallel fixpoint algorithm.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime = 0;

  auto gpu_data = bt::make_unique<MinimumData, bt::managed_allocator>(VERTICES, EDGES);
  gpu_data->edge = cpu_data->edge;
  init_distances(gpu_data->min_dist);

  int device = 0, maxShmemSz = 0, nProc = 0, maxBlocksPerSM = 0;
  cudaGetDevice(&device);
  int blocks = 2;
  cudaDeviceGetAttribute(&maxShmemSz, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cudaDeviceGetAttribute(&nProc, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, compute_min_grid, TPB, 0);
  blocks = std::min(blocks, maxBlocksPerSM * nProc);
  std::cout << "GPU Minimum on " << blocks <<  " blocks (max=" << maxBlocksPerSM  << "*" << nProc << "=" << maxBlocksPerSM * nProc << ")";
  std::cout << " with cooperative_groups";
  std::cout << " shared mem max=" << maxShmemSz/1024 << " kiB" << std::endl;
  std::cout << "\tinitial distances: ";
  for (size_t i = 0; i < gpu_data->min_dist.size(); ++i) {
        std::cout << "d[" << i << "]=" << gpu_data->min_dist[i] << " ";
  }
  std::cout << std::endl;

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
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "\tminimum distances: ";
  for (size_t i = 0; i < gpu_data->min_dist.size(); ++i) {
        std::cout << "d[" << i << "]=" << gpu_data->min_dist[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "\titerations: " << gpu_data->iterations;
  std::cout << "\telapsed: " << elapsedTime << " ms" << std::endl;

  for (unsigned int i=0; i < cpu_data->min_dist.size(); ++i) {
      if (cpu_data->min_dist[i] != gpu_data->min_dist[i]) {
          std::cerr << "distance mismatch!" << std::endl;
          return 1;
      }
  }
  std::cout << "distances match" << std::endl;
  return 0;
}
