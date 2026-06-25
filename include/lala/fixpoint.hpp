// Copyright 2022 Pierre Talbot

#ifndef LALA_FIXPOINT_FIXPOINT_HPP
#define LALA_FIXPOINT_FIXPOINT_HPP

#include "lala/lb.hpp"
#include "battery/memory.hpp"

#ifdef __CUDACC__
  #include <cooperative_groups.h>
#endif

namespace lala {

/** A simple form of sequential fixpoint computation based on Kleene fixpoint.
 * At each iteration, the deduction operations \f$ f_1, \ldots, f_n \f$ are simply composed by functional composition \f$ f = f_n \circ \ldots \circ f_1 \f$.
 * This strategy basically corresponds to the Gauss-Seidel iteration method. */
class GaussSeidelIteration {
public:
  CUDA void barrier() {}

  /** We iterate the function `f` `n` times: \f$ f(0); f(1); \ldots ; f(n); \f$
   * \param `n` the number of call to `f`.
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \return `true` if for some `i`, `f(i)` returned `true`, `false` otherwise.
  */
  template <class F>
  CUDA bool iterate(int n, const F& f) const {
    bool has_changed = false;
    for(int i = 0; i < n; ++i) {
      has_changed |= f(i);
    }
    return has_changed;
  }

  /** We execute `iterate(n, f)` until we reach a fixpoint or `must_stop()` returns `true`.
   * \param `n` the number of call to `f`.
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \param `bool must_stop()` returns `true` if we must stop early the fixpoint computation.
   * \param `has_changed` is set to `true` if we were not yet in a fixpoint.
   * \return The number of iterations required to reach a fixpoint or until `must_stop()` returns `true`.
  */
  template <class F, class StopFun, class M>
  CUDA int fixpoint(int n, const F& f, const StopFun& must_stop, LB<bool, M>& has_changed) {
    int iterations = 0;
    bool changed = true;
    while(changed && !must_stop()) {
      changed = iterate(n, f);
      has_changed.meet(changed);
      iterations++;
    }
    return iterations;
  }

  /** Same as `fixpoint` above without `has_changed`. */
  template <class F, class StopFun>
  CUDA int fixpoint(int n, const F& f, const StopFun& must_stop) {
    LB<bool> has_changed(false);
    return fixpoint(n, f, must_stop, has_changed);
  }

  /** Same as `fixpoint` above with `must_stop` always returning `false`. */
  template <class F, class M>
  CUDA int fixpoint(int n, const F& f, LB<bool, M>& has_changed) {
    return fixpoint(n, f, [](){ return false; }, has_changed);
  }

  /** Same as `fixpoint` above without `has_changed` and with `must_stop` always returning `false`. */
  template <class F>
  CUDA int fixpoint(int n, const F& f) {
    LB<bool> has_changed(false);
    return fixpoint(n, f, has_changed);
  }
};

#ifdef __CUDACC__

/** This fixpoint engine is parametrized by an iterator engine `I` providing a method `iterate(n,f)`, a barrier `barrier()` and a function `is_thread0()` returning `true` for a single thread.
 *  AsynchronousFixpoint provides a `fixpoint` function using `iterate` of `I`. */
template <class IteratorEngine>
class AsynchronousFixpoint {
  /** We do not use relaxed atomic because tearing is seemingly not possible in CUDA (according to information given by Nvidia engineers during a hackathon). */
  LB<bool> changed[3];
  LB<bool> stop[3];

  CUDA void reset() {
    if(is_thread0()) {
      changed[0] = true;
      changed[1] = false;
      changed[2] = false;
      for(int i = 0; i < 3; ++i) {
        stop[i] = false;
      }
    }
  }

public:
  CUDA INLINE bool is_thread0() {
    return static_cast<IteratorEngine*>(this)->is_thread0();
  }

  CUDA INLINE void barrier() {
    static_cast<IteratorEngine*>(this)->barrier();
  }

  template <class F>
  CUDA INLINE bool iterate(int n, const F& f) const {
    return static_cast<const IteratorEngine*>(this)->iterate(n, f);
  }

  /** We execute `I::iterate(n, f)` until we reach a fixpoint or `must_stop()` returns `true`.
   * \param `n` the number of call to `f`.
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \param `bool must_stop()` returns `true` if we must stop early the fixpoint computation. This function is called by the first thread only.
   * \param `has_changed` is set to `true` if we were not yet in a fixpoint.
   * \return The number of iterations required to reach a fixpoint or until `must_stop()` returns `true`.
  */
  template <class F, class StopFun, class M>
  CUDA int fixpoint(int n, const F& f, const StopFun& must_stop, LB<bool, M>& has_changed) {
    reset();
    barrier();
    int i;
    for(i = 1; changed[(i-1)%3] && !stop[(i-1)%3]; ++i) {
      changed[i%3].meet(iterate(n, f));  // changed[i] == false
      if(is_thread0()) {
        changed[(i+1)%3].join_top(); // reinitialize changed for the next iteration.
        stop[i%3].meet(must_stop());
      }
      barrier();
    }
    // It changes if we performed several iteration, or if the first iteration changed the abstract domain.
    if(is_thread0()) {
      has_changed.meet(changed[1] || i > 2);
    }
    return i - 1;
  }

  template <class F, class Iter, class StopFun>
  CUDA int fixpoint(int n, const F& f, const Iter& h, const StopFun& must_stop) {
    reset();
    barrier();
    int i;
    for(i = 1; changed[(i-1)%3] && !stop[(i-1)%3]; ++i) {
      changed[i%3].meet(iterate(n, f));
      if(is_thread0()) {
        changed[(i+1)%3].join_top(); // reinitialize changed for the next iteration.
        stop[i%3].meet(must_stop());
      }
      barrier();
      h();
      barrier();
    }
    return i - 1;
  }

  /** Same as `fixpoint` above without `has_changed`. */
  template <class F, class StopFun>
  CUDA INLINE int fixpoint(int n, const F& f, const StopFun& must_stop) {
    LB<bool> has_changed(false);
    return fixpoint(n, f, must_stop, has_changed);
  }

  /** Same as `fixpoint` above with `must_stop` always returning `false`. */
  template <class F, class M>
  CUDA INLINE int fixpoint(int n, const F& f, LB<bool, M>& has_changed) {
    return fixpoint(n, f, [](){ return false; }, has_changed);
  }

  /** Same as `fixpoint` above without `has_changed` and with `must_stop` always returning `false`. */
  template <class F>
  CUDA INLINE int fixpoint(int n, const F& f) {
    LB<bool> has_changed(false);
    return fixpoint(n, f, has_changed);
  }
};

/** A simple form of fixpoint computation based on Kleene fixpoint.
 * At each iteration, the functions \f$ f_1, \ldots, f_n \f$ are composed by parallel composition \f$ f = f_1 \| \ldots \| f_n \f$ meaning they are executed in parallel by different threads.
 * This is called an asynchronous iteration and it is due to (Cousot, Asynchronous iterative methods for solving a fixed point system of monotone equations in a complete lattice, 1977).
 * \tparam Group is a CUDA cooperative group class (note that we provide a more efficient implementation for block group below in `BlockAsynchronousIterationGPU`).
*/
template <class Group>
class AsynchronousIterationGPU : public AsynchronousFixpoint<AsynchronousIterationGPU<Group>> {
public:
  using group_type = Group;
private:
  Group group;

  CUDA void assert_cuda_arch() const {
    printf("AsynchronousIterationGPU must be used on the GPU device only.\n");
    assert(0);
  }

public:
  CUDA AsynchronousIterationGPU(const Group& group):
    group(group)
  {}

  CUDA void reset() const {}

  CUDA INLINE bool is_thread0() const {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
      return false;
  #else
    return group.thread_rank() == 0;
  #endif
  }

  /** A barrier used to synchronize the threads within the group between iterations. */
  CUDA INLINE void barrier() {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
  #else
    group.sync();
  #endif
  }

  /** The function `f` is called `n` times in parallel: \f$ f(0) \| f(1) \| \ldots \| f(n) \f$.
   * \param `n` the number of call to `f`.
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \return `true` if for some `i`, `f(i)` returned `true`, `false` otherwise.
  */
  template <class F>
  CUDA INLINE bool iterate(int n, const F& f) const {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return false;
  #else
    bool has_changed = false;
    for (int i = group.thread_rank(); i < n; i += group.num_threads()) {
      has_changed |= f(i);
    }
    return has_changed;
  #endif
  }
};

using GridAsynchronousFixpointGPU = AsynchronousIterationGPU<cooperative_groups::grid_group>;

/** An optimized version of `AsynchronousIterationGPU` when the fixpoint is computed on a single block.
 * We avoid the use of cooperative groups which take extra memory space.
 * `syncwarp` is a boolean to tell if `f` in `iterate` is syncing the warp or not, if it does and syncwarp is `true`, `iterate` will always iterate to a multiple of 32 threads by repeating the last index if necessary.
 */
template <bool syncwarp = false>
class BlockAsynchronousFixpointGPU : public AsynchronousFixpoint<BlockAsynchronousFixpointGPU<syncwarp>> {
private:
  CUDA void assert_cuda_arch() const {
    printf("BlockAsynchronousFixpointGPU must be used on the GPU device only.\n");
    assert(0);
  }

public:
  BlockAsynchronousFixpointGPU() = default;

  CUDA INLINE bool is_thread0() const {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
      return false;
  #else
    return threadIdx.x == 0;
  #endif
  }

  CUDA INLINE void barrier() {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
  #else
    __syncthreads();
  #endif
  }

  /** The function `f` is called `n` times in parallel: \f$ f(0) \| f(1) \| \ldots \| f(n-1) \f$.
   * If `n` is greater than the number of threads in the block, we perform a stride loop, without synchronization between two iterations.
   * \param `n` the number of calls to `f`.
   * \param `bool f(int i)` returns `true` if something has changed for `i`.
   * \return `true` if for some `i`, `f(i)` returned `true`, `false` otherwise.
  */
  template <class F>
  CUDA INLINE bool iterate(int n, const F& f) const {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
    return false;
  #else
    bool has_changed = false;
    int n2 = syncwarp && n != 0 ? max(n,n+(32-(n%32))) : n;
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
      has_changed |= f(syncwarp ? (i >= n ? n-1 : i) : i);  // |= because stride wants OR all f(i)
    }
    return has_changed;
  #endif
  }
};

#endif

} // namespace lala

#endif
