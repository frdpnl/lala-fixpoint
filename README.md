# Fixpoint Library

This library provides generic fixpoint engines optimized for GPU and the fixpoint characterization of several standard algorithms.
We intend to explore the parallel programming model explained in [AAAI 2022](https://ptal.github.io/papers/aaai2022.pdf).
It is essentially providing a small language to program parallel algorithms *correct-by-construction*.
We want to push this model and see how many standard algorithms we can reimplement as fixpoint functions over some lattice structures.

## Plan

- [x] Import and clean fixpoint algorithms from lala-core.
- [x] Algorithm: finding the minimum in a sequence.
- [x] Improve minimum algorithm to support computation over a grid.
- [ ] Improve minimum algorithm to support computation over a multi-grid.
- [ ] Algorithm: search an element in an unsorted list.
- [ ] Algorithm: union-find.
- [ ] Algorithm: breadth-first search algorithm.
- [ ] Algorithm: all-shortest paths algorithm (e.g. Floyd-Warshall).
- [ ] Combinatorial algorithm: [stable marriage problem](https://arxiv.org/pdf/2208.01370), redundant with Turbo?
- [ ] Combinatorial algorithm: [min-cut problems](https://arxiv.org/pdf/2512.18141), redundant with Turbo?
- [ ] Investigate: [stochastic gradient descent](https://proceedings.neurips.cc/paper_files/paper/2011/file/218a0aefd1d1a4be65601cc6ddc1520e-Paper.pdf).
- [ ] Investigate: [asynchronous algorithms](https://web.mit.edu/dimitrib/www/pdc.html)
- [ ] Benchmarking: Fixpoint algorithms VS optimized CUDA algorithms.

## Getting Started

To contribute, please fork the repository and send us your pull requests.
For the available lattices, you can check out [lala-interval](https://github.com/lattice-land/lala-interval/).

```
mkdir lattice-land
cd lattice-land
git clone <your fork of lala-fixpoint>
git clone git@github.com:lattice-land/cuda-battery.git
git clone git@github.com:lattice-land/lala-interval.git
cd <your fork of lala-fixpoint>
cmake --workflow --preset gpu-release-local --fresh
./build/gpu-release-local/minimum
```

You can add new binaries in `CMakeLists.txt`.

## Log
```
$ ./build/gpu-release-local/minimum
Data initialized. Starting GPU benchmarks.
CPU Minimum:
  minimum: -999999914
GPU Minimum with 1 block
  initially: ∞
  minimum: -999999914  iterations: 5  elapsed: 344.802 ms
GPU Minimum on 2 blocks (max: 8 * 108 = 864) with cooperative_groups shared mem max: 163 kiB
  initially: ∞
  minimum: -999999914  iterations: 6  elapsed: 191.321 ms
``

