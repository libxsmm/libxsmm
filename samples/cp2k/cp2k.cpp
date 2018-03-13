/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#if !defined(USE_HEADER_ONLY)
# include <libxsmm.h>
#else
# include <libxsmm_source.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#if defined(__MKL)
# include <mkl_service.h>
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE double
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
LIBXSMM_RETARGETABLE class LIBXSMM_RETARGETABLE lock_type {
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


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void init(libxsmm_blasint seed, REAL_TYPE *LIBXSMM_RESTRICT dst,
  libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  libxsmm_blasint i;
  for (i = 0; i < ncols; ++i) {
    libxsmm_blasint j = 0;
    for (; j < nrows; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = (REAL_TYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = (REAL_TYPE)seed;
    }
  }
}


template<typename T>
LIBXSMM_INLINE LIBXSMM_RETARGETABLE
void add(T *LIBXSMM_RESTRICT dst, const T *LIBXSMM_RESTRICT src, libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld_src = 0)
{
  const libxsmm_blasint ld = 0 == ld_src ? ncols : ld_src;
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
    typedef REAL_TYPE T;
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

    const libxsmm_blasint csize = m * n;
    if ((MAX_SIZE) < csize) {
      throw "The size M x N is exceeding MAX_SIZE!";
    }

    const libxsmm_blasint asize = m * k, bsize = k * n, aspace = LIBXSMM_ALIGNMENT / sizeof(T);
    const libxsmm_blasint s = 0 < r ? r : ((2ULL << 30) / ((asize + bsize) * sizeof(T))); // 2 GByte
    const libxsmm_blasint u = 0 < t ? t : static_cast<libxsmm_blasint>(libxsmm_isqrt_u64(s * CP2K_MIN_NLOCAL / CP2K_MIN_NPARALLEL));
    const size_t bwsize = static_cast<size_t>((s * (asize + bsize)/*load*/ + ((s + u - 1) / u) * csize * 2/*accumulate*/) * sizeof(T));
    const double gflops = 2.0 * s * m * n * k * 1E-9, scale = 1.0 / s;
    const char *const env_check = getenv("CHECK");
    const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));

    LIBXSMM_RETARGETABLE struct LIBXSMM_RETARGETABLE raii { // avoid std::vector (first-touch init. causes NUMA issue)
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
      init(42 + i, a + i * asize, m, k, m, scale);
      init(24 + i, b + i * bsize, k, n, k, scale);
    }

#if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload target(LIBXSMM_OFFLOAD_TARGET) in(a: length(s * asize)) in(b: length(s * bsize)) out(c: length(csize))
#endif
    {
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      // initialize LIBXSMM
      libxsmm_init();
#if !defined(LIBXSMM_OFFLOAD_TARGET)
      // some more setup similar to CP2K/intel branch
      libxsmm_set_gemm_auto_prefetch(LIBXSMM_X86_AVX512_MIC != libxsmm_get_target_archid() ? LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C : LIBXSMM_GEMM_PREFETCH_BL2_VIA_C);
#endif
      //libxsmm_set_dispatch_trylock(1);

      fprintf(stdout, "m=%lli n=%lli k=%lli size=%lli memory=%.1f MB (%s)\n\n",
        static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(s),
        1.0 * (s * (asize + bsize) * sizeof(T)) / (1 << 20), 8 == sizeof(T) ? "DP" : "SP");

      LIBXSMM_RETARGETABLE struct LIBXSMM_RETARGETABLE raii_expect { // avoid std::vector (first-touch init. causes NUMA issue)
        T *expect;
        explicit raii_expect(libxsmm_blasint size): expect(0 < size ? new T[static_cast<size_t>(size)] : 0) {}
        ~raii_expect() { delete[] expect; }
      } expect_buffer(LIBXSMM_FEQ(0, check) ? 0 : csize);
      T *const expect = (0 == expect_buffer.expect ? c : expect_buffer.expect);
      libxsmm_matdiff_info d, diff;
      const T zero = 0;

      // eventually JIT-compile the requested kernel
      const libxsmm_mmfunction<T> xmm(LIBXSMM_FLAGS, m, n, k, LIBXSMM_PREFETCH);

      memset(&diff, 0, sizeof(diff));
      { // LAPACK/BLAS3 (warmup BLAS Library)
        std::fill_n(expect, csize, zero);
#if defined(_OPENMP)
#       pragma omp parallel for CP2K_SCHEDULE
#endif
        for (libxsmm_blasint i = 0; i < s; i += u) {
          T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
          const T *ai = a + i * asize, *bi = b + i * bsize;
          for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            const T *const aij = ai + asize, *const bij = bi + bsize;
            libxsmm_blas_gemm(0/*transa*/, 0/*transb*/, m, n, k,
              0/*alpha*/, ai, 0/*lda*/, bi, 0/*ldb*/,
              0/*beta*/, tmp, 0/*ldc*/);
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
#       pragma omp parallel for CP2K_SCHEDULE
#endif
        for (libxsmm_blasint i = 0; i < s; i += u) {
          T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
          const T *ai = a + i * asize, *bi = b + i * bsize;
          for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            const T *const aij = ai + asize, *const bij = bi + bsize;
            // alternatively libxsmm_blas_gemm can be called (see above)
            LIBXSMM_BLAS_GEMM(LIBXSMM_FLAGS, m, n, k,
              LIBXSMM_ALPHA, ai, m, bi, k,
              LIBXSMM_BETA, tmp, m);
            ai = aij;
            bi = bij;
          }
          add(c, tmp, m, n); // atomic
        }
        const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
          fprintf(stdout, "\tcalls/s: %.0f Hz\n", s / duration);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (!LIBXSMM_FEQ(0, check) && EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(REAL_TYPE), m, n, expect, c, 0, 0, &d)) {
          fprintf(stdout, "\tdiff: L2abs=%f Linfo=%f\n", d.l2_abs, d.linf_abs);
          libxsmm_matdiff_reduce(&diff, &d);
        }
      }

      { // inline an optimized implementation
        fprintf(stdout, "Inlined...\n");
        std::fill_n(c, csize, zero);
        const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#       pragma omp parallel for CP2K_SCHEDULE
#endif
        for (libxsmm_blasint i = 0; i < s; i += u) {
          T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
          const T *ai = a + i * asize, *bi = b + i * bsize;
          for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            const T *const aij = ai + asize, *const bij = bi + bsize;
            LIBXSMM_INLINE_GEMM(LIBXSMM_FLAGS, m, n, k,
              LIBXSMM_ALPHA, ai, m, bi, k,
              LIBXSMM_BETA, tmp, m);
            ai = aij;
            bi = bij;
          }
          add(c, tmp, m, n); // atomic
        }
        const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
          fprintf(stdout, "\tcalls/s: %.0f Hz\n", s / duration);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (!LIBXSMM_FEQ(0, check) && EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(REAL_TYPE), m, n, expect, c, 0, 0, &d)) {
          fprintf(stdout, "\tdiff: L2abs=%f Linfo=%f\n", d.l2_abs, d.linf_abs);
          libxsmm_matdiff_reduce(&diff, &d);
        }
      }

      { // auto-dispatched
        fprintf(stdout, "Dispatched...\n");
        std::fill_n(c, csize, zero);
        const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#       pragma omp parallel for CP2K_SCHEDULE
#endif
        for (libxsmm_blasint i = 0; i < s; i += u) {
          T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
          const T *ai = a + i * asize, *bi = b + i * bsize;
          for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            const T *const aij = ai + asize, *const bij = bi + bsize;
            libxsmm_gemm(0/*transa*/, 0/*transb*/, m, n, k,
              0/*alpha*/, ai, 0/*lda*/, bi, 0/*ldb*/,
              0/*beta*/, tmp, 0/*ldc*/);
            ai = aij;
            bi = bij;
          }
          add(c, tmp, m, n); // atomic
        }
        const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
          fprintf(stdout, "\tcalls/s: %.0f Hz\n", s / duration);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (!LIBXSMM_FEQ(0, check) && EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(REAL_TYPE), m, n, expect, c, 0, 0, &d)) {
          fprintf(stdout, "\tdiff: L2abs=%f Linfo=%f\n", d.l2_abs, d.linf_abs);
          libxsmm_matdiff_reduce(&diff, &d);
        }
      }

      if (xmm) { // specialized routine
        fprintf(stdout, "Specialized...\n");
        std::fill_n(c, csize, zero);
        const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#       pragma omp parallel for CP2K_SCHEDULE
#endif
        for (libxsmm_blasint i = 0; i < s; i += u) {
          T tmp[MAX_SIZE] = { 0 }; // make sure that stacksize is covering the problem size
          const T *ai = a + i * asize, *bi = b + i * bsize;
          for (libxsmm_blasint j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            const T *const aij = ai + asize, *const bij = bi + bsize;
#if (0 != LIBXSMM_PREFETCH)
            xmm(ai, bi, tmp,
              LIBXSMM_GEMM_PREFETCH_A(aij + asize),
              LIBXSMM_GEMM_PREFETCH_B(bij + bsize),
              LIBXSMM_GEMM_PREFETCH_C(tmp));
#else
            xmm(ai, bi, tmp);
#endif
            ai = aij;
            bi = bij;
          }
          add(c, tmp, m, n); // atomic
        }
        const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
          fprintf(stdout, "\tcalls/s: %.0f Hz\n", s / duration);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (!LIBXSMM_FEQ(0, check) && EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(REAL_TYPE), m, n, expect, c, 0, 0, &d)) {
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
