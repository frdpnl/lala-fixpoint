# Fixpoint Library

This library provides generic fixpoint engines optimized for GPU and the fixpoint characterization of several standard algorithms.
We intend to explore the parallel programming model explained in [AAAI 2022](https://ptal.github.io/papers/aaai2022.pdf).
It is essentially providing a small language to program parallel algorithms *correct-by-construction*.
We want to push this model and see how many standard algorithms we can reimplement as fixpoint functions over some lattice structures.

## Plan

- [x] Import and clean fixpoint algorithms from lala-core.
- [x] Algorithm: finding the minimum in a sequence.
- [ ] Algorithm: search an element in an unsorted list.
- [ ] Algorithm: union-find.
- [ ] Algorithm: breadth-first search algorithm.
- [ ] Benchmarking: Fixpoint algorithms VS optimized CUDA algorithms.
