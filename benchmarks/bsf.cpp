// Copyright 2026 Pierre Talbot, Frederic Pinel

#include <limits>
#include <random>
#include <iostream>
#include <chrono>
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

void reset_dist(auto& d) {
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

static void cpu_bsf(const bt::unique_ptr<MinimumData, bt::managed_allocator>& data) 
{
  unsigned int vertices{static_cast<unsigned int>(data->min_dist.size())};
  reset_dist(data->min_dist);

  std::cout << "CPU Minimum" << std::endl;
  std::cout << "\tinitial distances: ";
  for (unsigned int i = 0; i < vertices; ++i) {
        std::cout << "d[" << i << "]=" << data->min_dist[i] << " ";
  }
  std::cout << std::endl;

  auto cpu_start = std::chrono::steady_clock::now();
  // On CPU, using a sequential fixpoint algorithm.
  GaussSeidelIteration fp_engine;
  data->iterations = fp_engine.fixpoint(
          data->edge.size()/2,  /* /2: edges are pairs of vertices */
          [&](int i) {  unsigned int left_ver = data->edge[2*i];
                        unsigned int right_ver = data->edge[2*i+1];
                        bool changed = false;
                        if (/* right_ver>0 && */ !data->min_dist[left_ver].is_top()) {
                            changed |= data->min_dist[right_ver].meet(1 + data->min_dist[left_ver]);
                        }
                        if (left_ver>0 && !data->min_dist[right_ver].is_top()) {
                            changed |= data->min_dist[left_ver].meet(1 + data->min_dist[right_ver]);
                        }
                        return changed;
                     }
  );
  auto cpu_stop = std::chrono::steady_clock::now();
  auto elapsed = cpu_stop - cpu_start;

  std::cout << "\tminimum distances: ";
  for (size_t i = 0; i < vertices; ++i) {
        std::cout << "d[" << i << "]=" << data->min_dist[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "\titerations: " << data->iterations << std::endl;
  std::cout << "\telapsed: " << elapsed << std::endl;  // TODO figure out conversions...
}

static void gpu_bsf(const bt::unique_ptr<MinimumData, bt::managed_allocator>& data, int B, int TpB) 
{
  unsigned int vertices{static_cast<unsigned int>(data->min_dist.size())};
  reset_dist(data->min_dist);

  // On GPU, using a grid-parallel fixpoint algorithm.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime{0};

  int device = 0, maxShmemSz = 0, nProc = 0, maxBlocksPerSM = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&maxShmemSz, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cudaDeviceGetAttribute(&nProc, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, compute_min_grid, TpB, 0);
  int blocks = std::min(B, maxBlocksPerSM * nProc);
  std::cout << "GPU Minimum on " << blocks <<  " blocks (max=" << maxBlocksPerSM  << "*" << nProc << "=" << maxBlocksPerSM * nProc << ")";
  std::cout << " with cooperative_groups";
  std::cout << " shared mem max=" << maxShmemSz/1024 << " kiB" << std::endl;
  std::cout << "\tinitial distances: ";
  for (size_t i = 0; i < vertices; ++i) {
        std::cout << "d[" << i << "]=" << data->min_dist[i] << " ";
  }
  std::cout << std::endl;

  dim3 dimBlock(TpB, 1, 1);
  dim3 dimGrid(blocks, 1, 1);  
  auto pdata = data.get();
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
  for (size_t i = 0; i < vertices; ++i) {
        std::cout << "d[" << i << "]=" << data->min_dist[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "\titerations: " << data->iterations;
  std::cout << "\telapsed: " << elapsedTime << " ms" << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 5) {
      std::cerr << "Usage: " << argv[0] << " vertices edges" << std::endl;
      return 1;
  }
  unsigned int edges{static_cast<unsigned int>(std::stoul(argv[1], nullptr, 10))};
  unsigned int vertices{static_cast<unsigned int>(std::stoul(argv[2], nullptr, 10))};
  unsigned int blocks{static_cast<unsigned int>(std::stoul(argv[3], nullptr, 10))};
  unsigned int threadsPerBlock{static_cast<unsigned int>(std::stoul(argv[4], nullptr, 10))};

  auto cpu_data = bt::make_unique<MinimumData, bt::managed_allocator>(vertices, edges);
  init_random_edges(cpu_data->edge, vertices -1);

  std::cout << "Edges " << edges << ", vertices " << vertices << " initialized." << std::endl;
  std::cout << "\tedges: ";
  // poor man's sort:
  for (unsigned int v = 0; v < vertices; ++v) {
      std::cout << v << "->";
      for (unsigned int left = 0; left < edges; ++left) {
          if (cpu_data->edge[2*left] == v) {
            std::cout << cpu_data->edge[2*left+1];
            std::cout << ",";
          }
      }
      std::cout << " ";
  }
  std::cout << std::endl;

  cpu_bsf(cpu_data);

  auto gpu_data = bt::make_unique<MinimumData, bt::managed_allocator>(vertices, edges);
  gpu_data->edge = cpu_data->edge;
  gpu_bsf(gpu_data, blocks, threadsPerBlock);

  for (unsigned int i=0; i < vertices; ++i) {
      if (cpu_data->min_dist[i] != gpu_data->min_dist[i]) {
          std::cerr << "distance mismatch!" << std::endl;
          return 1;
      }
  }
  std::cout << "distances match" << std::endl;
  return 0;
}
