/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD)
# pragma offload_attribute(push,target(mic))
#endif

#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>

#if defined(USE_MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif

#if defined(_OPENMP)
# include <omp.h>
#endif

#if defined(LIBXSMM_OFFLOAD)
# pragma offload_attribute(pop)
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
// Kind of thread-private data
#define CP2K_THREADPRIVATE 1
// enable result validation
#define CP2K_CHECK


#if defined(_OPENMP) && defined(CP2K_SYNCHRONIZATION) && (1 < (CP2K_SYNCHRONIZATION))
LIBXSMM_TARGET(mic) class LIBXSMM_TARGET(mic) lock_type {
public:
  lock_type() {
    for (int i = 0; i < (CP2K_SYNCHRONIZATION); ++i) omp_init_lock(m_lock + i);
  }
  
  ~lock_type() {
    for (int i = 0; i < (CP2K_SYNCHRONIZATION); ++i) omp_destroy_lock(m_lock + i);
  }

public:
  void acquire(const void* address) {
    const uintptr_t id = reinterpret_cast<uintptr_t>(address) / (LIBXSMM_ALIGNED_MAX);
    // non-pot: omp_set_lock(m_lock + id % CP2K_SYNCHRONIZATION);
    omp_set_lock(m_lock + LIBXSMM_MOD(id, CP2K_SYNCHRONIZATION));
  }

  void release(const void* address) {
    const uintptr_t id = reinterpret_cast<uintptr_t>(address) / (LIBXSMM_ALIGNED_MAX);
    // non-pot: omp_unset_lock(m_lock + id % CP2K_SYNCHRONIZATION);
    omp_unset_lock(m_lock + LIBXSMM_MOD(id, CP2K_SYNCHRONIZATION));
  }

private:
  omp_lock_t m_lock[CP2K_SYNCHRONIZATION];
} lock;
#endif


template<int Seed>
struct LIBXSMM_TARGET(mic) init {
  template<typename T> init(T *LIBXSMM_RESTRICT dst, int m, int n, int ld = 0) {
    const int ldx = 0 == ld ? LIBXSMM_LD(m, n) : ld;
    for (int i = 0; i < LIBXSMM_LD(n, m); ++i) {
      for (int j = 0; j < LIBXSMM_LD(m, n); ++j) {
        // initialize similar to CP2K's (libsmm_acc) benchmark driver
        dst[i*ldx+j] = static_cast<T>(i * ldx + j + n + Seed);
      }
    }
  }
};


template<>
struct LIBXSMM_TARGET(mic) init<0> {
  template<typename T> init(T *LIBXSMM_RESTRICT dst, int m, int n, int ld = 0) {
    static const double scale = 1.0 / RAND_MAX;
    const int ldx = 0 == ld ? LIBXSMM_LD(m, n) : ld;
    for (int i = 0; i < LIBXSMM_LD(n, m); ++i) {
      for (int j = 0; j < LIBXSMM_LD(m, n); ++j) {
        // initialize every value using a normalized random number [-1, +1]
        dst[i*ldx+j] = static_cast<T>(scale * (2 * std::rand() - RAND_MAX));
      }
    }
  }
};


template<typename T>
LIBXSMM_TARGET(mic) void add(T *LIBXSMM_RESTRICT dst, const T *LIBXSMM_RESTRICT src, int m, int n, int ld_src = 0)
{
  const int ld = 0 == ld_src ? LIBXSMM_LD(m, n) : ld_src;
#if defined(_OPENMP) && defined(CP2K_SYNCHRONIZATION) && (0 < (CP2K_SYNCHRONIZATION))
# if (1 == (CP2K_SYNCHRONIZATION))
# pragma omp critical(smmadd)
# else
  lock.acquire(dst);
# endif
#endif
  {
    LIBXSMM_ASSUME_ALIGNED_STORES(src);
    for (int i = 0; i < LIBXSMM_LD(n, m); ++i) {
      LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_LD(LIBXSMM_MAX_M, LIBXSMM_MAX_N), LIBXSMM_LD(LIBXSMM_AVG_M, LIBXSMM_AVG_N))
      for (int j = 0; j < LIBXSMM_LD(m, n); ++j) {
        const T value = src[i*ld+j];
#if defined(_OPENMP) && (!defined(CP2K_SYNCHRONIZATION) || (0 == (CP2K_SYNCHRONIZATION)))
#       pragma omp atomic
#endif
        dst[i*LIBXSMM_LD(m,n)+j] += value;
      }
    }
  }
#if defined(_OPENMP) && defined(CP2K_SYNCHRONIZATION) && (1 < (CP2K_SYNCHRONIZATION))
  lock.release(dst);
#endif
}


template<typename T>
LIBXSMM_TARGET(mic) double max_diff(const T *LIBXSMM_RESTRICT result, const T *LIBXSMM_RESTRICT expect, int m, int n, int ld = 0)
{
  const int ldx = 0 == ld ? LIBXSMM_LD(m, n) : ld;
  double diff = 0;
  for (int i = 0; i < LIBXSMM_LD(n, m); ++i) {
    for (int j = 0; j < LIBXSMM_LD(m, n); ++j) {
      const int k = i * ldx + j;
      diff = std::max(diff, std::abs(static_cast<double>(result[k]) - static_cast<double>(expect[k])));
    }
  }
  return diff;
}


int main(int argc, char* argv[])
{
  try {
    typedef double T;
    const int m = 1 < argc ? std::atoi(argv[1]) : 23;
    const int q = ((1ULL << 30) / (3 * m * m * sizeof(T)));
    const int r = 2 < argc ? (0 < std::atoi(argv[2]) ? std::atoi(argv[2]) : ('+' == *argv[2]
      ? (q << std::strlen(argv[2])) : ('-' == *argv[2]
      ? (q >> std::strlen(argv[2])) : 0))) : 0;
    const int t = 3 < argc ? (0 < std::atoi(argv[3]) ? std::atoi(argv[3]) : ('+' == *argv[3]
      ? ((CP2K_MIN_NLOCAL) << std::strlen(argv[3])) : ('-' == *argv[3]
      ? ((CP2K_MIN_NLOCAL) >> std::strlen(argv[3])) : -1))) : -1;
    const int n = 4 < argc ? std::atoi(argv[4]) : m;
    const int k = 5 < argc ? std::atoi(argv[5]) : m;

    if ((LIBXSMM_MAX_SIZE) < size_t(m * n)) {
      throw std::runtime_error("The size M x N is exceeding LIBXSMM_MAX_SIZE!");
    }

    const int asize = m * k, bsize = k * n, aspace = (LIBXSMM_ALIGNMENT) / sizeof(T);
    const int ldc = LIBXSMM_LDC(T, int, m, n), csize = LIBXSMM_LD(n, m) * ldc;
    const int s = 0 < r ? r : ((1ULL << 30) / ((asize + bsize + csize) * sizeof(T)));
    const int u = 0 < t ? t : static_cast<int>(std::sqrt(static_cast<double>(s) * CP2K_MIN_NLOCAL / CP2K_MIN_NPARALLEL) + 0.5);
#if defined(_OPENMP)
    const size_t bwsize = (s * (asize + bsize)/*load*/ + ((s + u - 1) / u) * csize * 2/*accumulate*/) * sizeof(T);
    const double gflops = 2.0 * s * m * n * k * 1E-9;
#endif

    LIBXSMM_TARGET(mic) struct LIBXSMM_TARGET(mic) raii { // avoid std::vector (first-touch init. causes NUMA issue)
      T *a, *b, *c;
      raii(int s, int asize, int bsize, int csize, int aspace)
        : a(new T[s*asize+aspace-1]), b(new T[s*bsize+aspace-1]), c(new T[csize])
      {}
      ~raii() { delete[] a; delete[] b; delete[] c; }
    } buffer(s, asize, bsize, csize, aspace);
    T *const a = LIBXSMM_ALIGN(T*, buffer.a, LIBXSMM_ALIGNED_MAX);
    T *const b = LIBXSMM_ALIGN(T*, buffer.b, LIBXSMM_ALIGNED_MAX);
    T * /*const*/ c = buffer.c; // no alignment, but thread-local array will be aligned

#if defined(_OPENMP)
#   pragma omp parallel for CP2K_SCHEDULE
#endif
    for (int i = 0; i < s; ++i) {
      init<42>(a + i * asize, m, k);
      init<24>(b + i * bsize, k, n);
    }

#if defined(LIBXSMM_OFFLOAD)
#   pragma offload target(mic) in(a: length(s * asize)) in(b: length(s * bsize)) out(c: length(csize))
#endif
    {
#if defined(CP2K_THREADPRIVATE) && defined(_OPENMP)
# if 1 == (CP2K_THREADPRIVATE) // native OpenMP TLS
      LIBXSMM_TARGET(mic) LIBXSMM_ALIGNED(static T tmp[LIBXSMM_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
#     pragma omp threadprivate(tmp)
# else
      LIBXSMM_TARGET(mic) LIBXSMM_ALIGNED(static LIBXSMM_TLS T tmp[LIBXSMM_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
# endif
#else // without OpenMP nothing needs to be thread-local due to a single-threaded program
      LIBXSMM_TARGET(mic) LIBXSMM_ALIGNED(static T tmp[LIBXSMM_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
#endif
#if defined(USE_MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
      mkl_enable_instructions(MKL_ENABLE_AVX512_MIC);
#endif
      fprintf(stdout, "m=%i n=%i k=%i ldc=%i (%s) size=%i batch=%i memory=%.f MB\n\n",
        m, n, k, ldc, 0 != (LIBXSMM_ROW_MAJOR) ? "row-major" : "column-major", s, u,
        1.0 * (s * (asize + bsize + csize) * sizeof(T)) / (1 << 20));

#if defined(CP2K_CHECK)
      LIBXSMM_TARGET(mic) struct LIBXSMM_TARGET(mic) raii { // avoid std::vector (first-touch init. causes NUMA issue)
        T *expect;
        raii(int csize): expect(new T[csize]) {}
        ~raii() { delete[] expect; }
      } buffer(csize);
      T *const expect = buffer.expect;
#endif

      { // LAPACK/BLAS3 (fallback/reference)
        fprintf(stdout, "LAPACK/BLAS...\n");
        std::fill_n(c, csize, 0);
#if defined(_OPENMP)
        double start = 0;
#       pragma omp parallel
        {
#         pragma omp master
          start = omp_get_wtime();
#         pragma omp for CP2K_SCHEDULE
#endif
          for (int i = 0; i < s; i += u) {
#if !defined(CP2K_THREADPRIVATE)
            LIBXSMM_ALIGNED(T tmp[LIBXSMM_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
#endif
            for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
            for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
              libxsmm_blasmm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
            }
            add(c, tmp, m, n, ldc); // atomic
          }
#if defined(_OPENMP)
        }
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(CP2K_CHECK)
        std::copy(c, c + csize, expect);
#endif
      }

      { // inline an optimized implementation
        fprintf(stdout, "Inlined...\n");
        std::fill_n(c, csize, 0);
#if defined(_OPENMP)
        double start = 0;
#       pragma omp parallel
        {
#         pragma omp master
          start = omp_get_wtime();
#         pragma omp for CP2K_SCHEDULE
#endif
          for (int i = 0; i < s; i += u) {
#if !defined(CP2K_THREADPRIVATE)
            LIBXSMM_ALIGNED(T tmp[LIBXSMM_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
#endif
            for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
            for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
              libxsmm_imm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
            }
            add(c, tmp, m, n, ldc); // atomic
          }
#if defined(_OPENMP)
        }
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(CP2K_CHECK)
        fprintf(stdout, "\tdiff=%f\n", max_diff(c, expect, m, n, ldc));
#endif
      }

      { // auto-dispatched
        fprintf(stdout, "Dispatched...\n");
        std::fill_n(c, csize, 0);
#if defined(_OPENMP)
        double start = 0;
#       pragma omp parallel
        {
#         pragma omp master
          start = omp_get_wtime();
#         pragma omp for CP2K_SCHEDULE
#endif
          for (int i = 0; i < s; i += u) {
#if !defined(CP2K_THREADPRIVATE)
            LIBXSMM_ALIGNED(T tmp[LIBXSMM_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
#endif
            for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
            for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
              libxsmm_mm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
            }
            add(c, tmp, m, n, ldc); // atomic
          }
#if defined(_OPENMP)
        }
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(CP2K_CHECK)
        fprintf(stdout, "\tdiff=%f\n", max_diff(c, expect, m, n, ldc));
#endif
      }

      const libxsmm_mm_dispatch<T> xmm(m, n, k);
      if (xmm) { // specialized routine
        fprintf(stdout, "Specialized...\n");
        std::fill_n(c, csize, 0);
#if defined(_OPENMP)
        double start = 0;
#       pragma omp parallel
        {
#         pragma omp master
          start = omp_get_wtime();
#         pragma omp for CP2K_SCHEDULE
#endif
          for (int i = 0; i < s; i += u) {
#if !defined(CP2K_THREADPRIVATE)
            LIBXSMM_ALIGNED(T tmp[LIBXSMM_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
#endif
            for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
            for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
              xmm(&a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
            }
            add(c, tmp, m, n, ldc); // atomic
          }
#if defined(_OPENMP)
        }
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(CP2K_CHECK)
        fprintf(stdout, "\tdiff=%f\n", max_diff(c, expect, m, n, ldc));
#endif
      }

      fprintf(stdout, "Finished\n");
    }
  }
  catch(const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
  }
  catch(...) {
    fprintf(stderr, "Error: unknown exception caught!\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
