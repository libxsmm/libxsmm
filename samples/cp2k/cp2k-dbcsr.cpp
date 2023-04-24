/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#if !defined(USE_HEADER_ONLY)
# include <libxsmm.h>
#else
# include <libxsmm_source.h>
#endif
#include <utils/libxsmm_utils.h>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(REALTYPE)
# define REALTYPE double
#endif

#if (LIBXSMM_EQUAL(REALTYPE, float) || LIBXSMM_EQUAL(REALTYPE, double)) \
  && !defined(MKL_DIRECT_CALL_SEQ) && !defined(MKL_DIRECT_CALL)
LIBXSMM_BLAS_SYMBOL_DECL(REALTYPE, gemm)
#endif

#if !defined(MAX_SIZE)
# define MAX_SIZE ((LIBXSMM_MAX_M) * (LIBXSMM_MAX_N))
#endif

/** >1: number of locks, =1: omp critical, =0: atomic */
#define CP2K_SYNCHRONIZATION 0
// ensures sufficient parallel slack
#define CP2K_MIN_NPARALLEL 240
// ensures amortized atomic overhead
#define CP2K_MIN_NLOCAL 160
// OpenMP schedule policy (and chunk size)
#if defined(__MIC__)
# define CP2K_SCHEDULE schedule(static,1)
#else
# define CP2K_SCHEDULE
#endif


#if defined(_OPENMP) && defined(CP2K_SYNCHRONIZATION) && (1 < (CP2K_SYNCHRONIZATION))
class lock_type {
public:
  lock_type() {
    for (int i = 0; i < (CP2K_SYNCHRONIZATION); ++i) omp_init_lock(m_lock + i);
  }
  ~lock_type() {
    for (int i = 0; i < (CP2K_SYNCHRONIZATION); ++i) omp_destroy_lock(m_lock + i);
  }
public:
  void acquire(const void* address) {
    omp_set_lock(m_lock + LIBXSMM_FOLD2(address, LIBXSMM_ALIGNMENT, CP2K_SYNCHRONIZATION));
  }
  void release(const void* address) {
    omp_unset_lock(m_lock + LIBXSMM_FOLD2(address, LIBXSMM_ALIGNMENT, CP2K_SYNCHRONIZATION));
  }
private:
  omp_lock_t m_lock[CP2K_SYNCHRONIZATION];
} lock;
#endif


template<typename T>
LIBXSMM_INLINE
void add(T *LIBXSMM_RESTRICT dst, const T *LIBXSMM_RESTRICT src, libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld_src = 0)
{
  const libxsmm_blasint ld = (0 == ld_src ? ncols : ld_src);
#if defined(_OPENMP) && defined(CP2K_SYNCHRONIZATION) && (0 < (CP2K_SYNCHRONIZATION))
# if (1 == (CP2K_SYNCHRONIZATION))
# pragma omp critical(smmadd)
# else
  lock.acquire(dst);
# endif
#endif
  {
    for (libxsmm_blasint i = 0; i < nrows; ++i) {
      LIBXSMM_PRAGMA_UNROLL
      for (libxsmm_blasint j = 0; j < ncols; ++j) {
        const T value = src[i*ld+j];
#if defined(_OPENMP) && (!defined(CP2K_SYNCHRONIZATION) || (0 == (CP2K_SYNCHRONIZATION)))
#       pragma omp atomic
#endif
        dst[i*ncols+j] += value;
      }
    }
  }
#if defined(_OPENMP) && defined(CP2K_SYNCHRONIZATION) && (1 < (CP2K_SYNCHRONIZATION))
  lock.release(dst);
#endif
}


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  try {
    typedef REALTYPE T;
    const libxsmm_blasint m = 1 < argc ? std::atoi(argv[1]) : 23;
    const libxsmm_blasint q = ((1ULL << 30) / (3 * m * m * sizeof(T)));
    const libxsmm_blasint r = 2 < argc ? (0 < std::atoi(argv[2]) ? std::atoi(argv[2]) : ('+' == *argv[2]
      ? (q << std::strlen(argv[2])) : ('-' == *argv[2]
      ? (q >> std::strlen(argv[2])) : 0))) : 0;
    const libxsmm_blasint t = 3 < argc ? (0 < std::atoi(argv[3]) ? std::atoi(argv[3]) : ('+' == *argv[3]
      ? ((CP2K_MIN_NLOCAL) << std::strlen(argv[3])) : ('-' == *argv[3]
      ? ((CP2K_MIN_NLOCAL) >> std::strlen(argv[3])) : -1))) : -1;
    const libxsmm_blasint k = 5 < argc ? std::atoi(argv[5]) : m;
    const libxsmm_blasint n = 4 < argc ? std::atoi(argv[4]) : k;
    const char transa = 'N', transb = 'N';
    const REALTYPE alpha = 1, beta = 1;

    const libxsmm_blasint csize = m * n;
    if ((MAX_SIZE) < csize) {
      throw "The size M x N is exceeding MAX_SIZE!";
    }

    const libxsmm_blasint asize = m * k, bsize = k * n, aspace = LIBXSMM_ALIGNMENT / sizeof(T);
    const libxsmm_blasint s = 0 < r ? r : ((2ULL << 30) / ((asize + bsize) * sizeof(T))); // 2 GByte
    const libxsmm_blasint u = 0 < t ? t : static_cast<libxsmm_blasint>(libxsmm_isqrt_u64(s * CP2K_MIN_NLOCAL / CP2K_MIN_NPARALLEL));
    const size_t bwsize = static_cast<size_t>((s * (asize + bsize)/*load*/ + LIBXSMM_UPDIV(s, u) * csize * 2/*accumulate*/) * sizeof(T));
    const double gflops = 2.0 * s * m * n * k * 1E-9, scale = 1.0 / s;
    const char ops[] = "FLOPS";
    const char *const env_check = getenv("CHECK");
    const double check = LIBXSMM_ABS(NULL == env_check ? 0 : atof(env_check));

    struct raii { // avoid std::vector (first-touch init. causes NUMA issue)
      T *a, *b, *c;
      raii(libxsmm_blasint asize_, libxsmm_blasint bsize_, libxsmm_blasint csize_)
        : a(new T[static_cast<size_t>(asize_)]), b(new T[static_cast<size_t>(bsize_)])
        , c(new T[static_cast<size_t>(csize_)]) {}
      ~raii() { delete[] a; delete[] b; delete[] c; }
    } buffer(s * asize + aspace - 1, s * bsize + aspace - 1, csize);
    T *const a = LIBXSMM_ALIGN(buffer.a, LIBXSMM_ALIGNMENT);
    T *const b = LIBXSMM_ALIGN(buffer.b, LIBXSMM_ALIGNMENT);
    T * /*const*/ c = buffer.c; // no alignment, but thread-local array will be aligned

#if defined(_OPENMP)
#   pragma omp parallel for
#endif
    for (libxsmm_blasint i = 0; i < s; ++i) {
      LIBXSMM_MATINIT(REALTYPE, 42 + i, a + i * asize, m, k, m, scale);
      LIBXSMM_MATINIT(REALTYPE, 24 + i, b + i * bsize, k, n, k, scale);
    }

    // initialize LIBXSMM
    libxsmm_init();
    // some more setup similar to CP2K/intel branch
    libxsmm_set_gemm_auto_prefetch(LIBXSMM_X86_AVX512_MIC != libxsmm_get_target_archid() ? LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C : LIBXSMM_GEMM_PREFETCH_BL2_VIA_C);
    //libxsmm_set_dispatch_trylock(1);

    fprintf(stdout, "m=%lli n=%lli k=%lli size=%lli memory=%.1f MB (%s)\n\n",
      static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(s),
      1.0 * (s * (asize + bsize) * sizeof(T)) / (1 << 20), 8 == sizeof(T) ? "DP" : "SP");

    struct raii_expect { // avoid std::vector (first-touch init. causes NUMA issue)
      T *expect;
      explicit raii_expect(libxsmm_blasint size): expect(0 < size ? new T[static_cast<size_t>(size)] : 0) {}
      ~raii_expect() { delete[] expect; }
    } expect_buffer(LIBXSMM_FEQ(0, check) ? 0 : csize);
    T *const expect = (0 == expect_buffer.expect ? c : expect_buffer.expect);
    libxsmm_matdiff_info d, diff;
    const T zero = 0;

    // eventually JIT-compile the requested kernel
    const libxsmm_mmfunction<T> xmm(LIBXSMM_GEMM_FLAGS(transa, transb), m, n, k);

    libxsmm_matdiff_clear(&diff);
    { // LAPACK/BLAS3 (warmup BLAS Library)
      std::fill_n(expect, csize, zero);
#if defined(_OPENMP)
#     pragma omp parallel for CP2K_SCHEDULE
#endif
      for (libxsmm_blasint i = 0; i < s; i += u) {
        T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
        const T *ai = a + i * asize, *bi = b + i * bsize;
        for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
          const T *const aij = ai + asize, *const bij = bi + bsize;
          LIBXSMM_GEMM_SYMBOL(REALTYPE)(&transa, &transb, &m, &n, &k,
            &alpha, ai, &m, bi, &k, &beta, tmp, &m);
          ai = aij;
          bi = bij;
        }
        add(expect, tmp, m, n); // atomic
      }
    }

    { // LAPACK/BLAS3 (reference)
      fprintf(stdout, "LAPACK/BLAS...\n");
      std::fill_n(c, csize, zero);
      const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel for CP2K_SCHEDULE
#endif
      for (libxsmm_blasint i = 0; i < s; i += u) {
        T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
        const T *ai = a + i * asize, *bi = b + i * bsize;
        for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
          const T *const aij = ai + asize, *const bij = bi + bsize;
          LIBXSMM_GEMM_SYMBOL(REALTYPE)(&transa, &transb, &m, &n, &k,
            &alpha, ai, &m, bi, &k, &beta, tmp, &m);
          ai = aij;
          bi = bij;
        }
        add(c, tmp, m, n); // atomic
      }
      const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
        fprintf(stdout, "\tcalls/s: %.0f Hz\n", s / duration);
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      if (!LIBXSMM_FEQ(0, check) && EXIT_SUCCESS == libxsmm_matdiff(&d, LIBXSMM_DATATYPE(REALTYPE), m, n, expect, c, 0, 0)) {
        fprintf(stdout, "\tdiff: L2abs=%f Linfo=%f\n", d.l2_abs, d.linf_abs);
        libxsmm_matdiff_reduce(&diff, &d);
      }
    }

    { // inline an optimized implementation
      fprintf(stdout, "Inlined...\n");
      std::fill_n(c, csize, zero);
      const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel for CP2K_SCHEDULE
#endif
      for (libxsmm_blasint i = 0; i < s; i += u) {
        T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
        const T *ai = a + i * asize, *bi = b + i * bsize;
        for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
          const T *const aij = ai + asize, *const bij = bi + bsize;
          LIBXSMM_INLINE_XGEMM(REALTYPE, REALTYPE, &transa, &transb, &m, &n, &k,
            &alpha, ai, &m, bi, &k, &beta, tmp, &m);
          ai = aij;
          bi = bij;
        }
        add(c, tmp, m, n); // atomic
      }
      const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
        fprintf(stdout, "\tcalls/s: %.0f Hz\n", s / duration);
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      if (!LIBXSMM_FEQ(0, check) && EXIT_SUCCESS == libxsmm_matdiff(&d, LIBXSMM_DATATYPE(REALTYPE), m, n, expect, c, 0, 0)) {
        fprintf(stdout, "\tdiff: L2abs=%f Linfo=%f\n", d.l2_abs, d.linf_abs);
        libxsmm_matdiff_reduce(&diff, &d);
      }
    }

    { // auto-dispatched
      fprintf(stdout, "Dispatched...\n");
      std::fill_n(c, csize, zero);
      const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel for CP2K_SCHEDULE
#endif
      for (libxsmm_blasint i = 0; i < s; i += u) {
        T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
        const T *ai = a + i * asize, *bi = b + i * bsize;
        for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
          const T *const aij = ai + asize, *const bij = bi + bsize;
          libxsmm_gemm(&transa, &transb, m, n, k,
            &alpha, ai, &m, bi, &k, &beta, tmp, &m);
          ai = aij;
          bi = bij;
        }
        add(c, tmp, m, n); // atomic
      }
      const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
        fprintf(stdout, "\tcalls/s: %.0f Hz\n", s / duration);
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      if (!LIBXSMM_FEQ(0, check) && EXIT_SUCCESS == libxsmm_matdiff(&d, LIBXSMM_DATATYPE(REALTYPE), m, n, expect, c, 0, 0)) {
        fprintf(stdout, "\tdiff: L2abs=%f Linfo=%f\n", d.l2_abs, d.linf_abs);
        libxsmm_matdiff_reduce(&diff, &d);
      }
    }

    if (xmm) { // specialized routine
      fprintf(stdout, "Specialized...\n");
      std::fill_n(c, csize, zero);
      const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel for CP2K_SCHEDULE
#endif
      for (libxsmm_blasint i = 0; i < s; i += u) {
        T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
        const T *ai = a + i * asize, *bi = b + i * bsize;
        for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
          const T *const aij = ai + asize, *const bij = bi + bsize;
          xmm(ai, bi, tmp, aij + asize, bij + bsize, tmp);
          ai = aij;
          bi = bij;
        }
        add(c, tmp, m, n); // atomic
      }
      const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
        fprintf(stdout, "\tcalls/s: %.0f Hz\n", s / duration);
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      if (!LIBXSMM_FEQ(0, check) && EXIT_SUCCESS == libxsmm_matdiff(&d, LIBXSMM_DATATYPE(REALTYPE), m, n, expect, c, 0, 0)) {
        fprintf(stdout, "\tdiff: L2abs=%f Linfo=%f\n", d.l2_abs, d.linf_abs);
        libxsmm_matdiff_reduce(&diff, &d);
      }
    }

    // finalize LIBXSMM
    libxsmm_finalize();
    fprintf(stdout, "Finished\n");

    if (!LIBXSMM_FEQ(0, check)) {
      if (check < 100.0 * diff.normf_rel) {
        fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
        result = EXIT_FAILURE;
      }
    }
  }
  catch(const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    result = EXIT_FAILURE;
  }
  catch(const char* message) {
    fprintf(stderr, "Error: %s\n", message);
    result = EXIT_FAILURE;
  }
  catch(...) {
    fprintf(stderr, "Error: unknown exception caught!\n");
    result = EXIT_FAILURE;
  }

  return result;
}
