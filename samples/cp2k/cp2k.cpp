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

#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif

#if defined(_OPENMP)
# include <omp.h>
#endif

#if defined(LIBXSMM_OFFLOAD)
# pragma offload_attribute(pop)
#endif

#define CP2K_MAX_SIZE (64 * 64)
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
    omp_set_lock(m_lock + LIBXSMM_HASH2(address, LIBXSMM_ALIGNED_MAX, CP2K_SYNCHRONIZATION));
  }

  void release(const void* address) {
    omp_unset_lock(m_lock + LIBXSMM_HASH2(address, LIBXSMM_ALIGNED_MAX, CP2K_SYNCHRONIZATION));
  }

private:
  omp_lock_t m_lock[CP2K_SYNCHRONIZATION];
} lock;
#endif


template<int Seed>
struct LIBXSMM_TARGET(mic) init {
  template<typename T> init(T *LIBXSMM_RESTRICT dst, int nrows, int ncols, int n = 0, int ld = 0) {
    const int ldx = 0 == ld ? ncols : ld;
    const double minval = n + Seed, addval = (nrows - 1) * ldx + (ncols - 1);
    const double shift = 0.5 * addval + minval, norm = 0 != addval ? (1.0 / addval) : 1.0;
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        const double value = static_cast<double>(i * ldx + j + minval);
        dst[i*ldx+j] = static_cast<T>(norm * (value - shift));
      }
    }
  }
};


template<typename T>
LIBXSMM_TARGET(mic) void add(T *LIBXSMM_RESTRICT dst, const T *LIBXSMM_RESTRICT src, int nrows, int ncols, int ld_src = 0)
{
  const int ld = 0 == ld_src ? ncols : ld_src;
#if defined(_OPENMP) && defined(CP2K_SYNCHRONIZATION) && (0 < (CP2K_SYNCHRONIZATION))
# if (1 == (CP2K_SYNCHRONIZATION))
# pragma omp critical(smmadd)
# else
  lock.acquire(dst);
# endif
#endif
  {
    LIBXSMM_ASSUME_ALIGNED_STORES(src);
    for (int i = 0; i < nrows; ++i) {
      LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_LD(LIBXSMM_MAX_M, LIBXSMM_MAX_N), LIBXSMM_LD(LIBXSMM_AVG_M, LIBXSMM_AVG_N))
      for (int j = 0; j < ncols; ++j) {
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


template<typename T>
LIBXSMM_TARGET(mic) double max_diff(const T *LIBXSMM_RESTRICT result, const T *LIBXSMM_RESTRICT expect, int nrows, int ncols, int ld = 0)
{
  const int ldx = 0 == ld ? ncols : ld;
  double diff = 0;
  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
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

    const int csize = m * n;
    if ((CP2K_MAX_SIZE) < csize) {
      throw std::runtime_error("The size M x N is exceeding CP2K_MAX_SIZE!");
    }

    const int asize = m * k, bsize = k * n, aspace = (LIBXSMM_ALIGNED_MAX) / sizeof(T);
    const int ldc = LIBXSMM_ALIGN_STORES(LIBXSMM_LD(m, n), sizeof(T));
    const int s = 0 < r ? r : ((2ULL << 30) / ((asize + bsize) * sizeof(T))); // 2 GByte
    const int u = 0 < t ? t : static_cast<int>(std::sqrt(static_cast<double>(s) * CP2K_MIN_NLOCAL / CP2K_MIN_NPARALLEL) + 0.5);
#if defined(_OPENMP)
    const size_t bwsize = (s * (asize + bsize)/*load*/ + ((s + u - 1) / u) * csize * 2/*accumulate*/) * sizeof(T);
    const double gflops = 2.0 * s * m * n * k * 1E-9;
#endif

    LIBXSMM_TARGET(mic) struct LIBXSMM_TARGET(mic) raii { // avoid std::vector (first-touch init. causes NUMA issue)
      T *a, *b, *c;
      raii(int asize, int bsize, int csize): a(new T[asize]), b(new T[bsize]), c(new T[csize]) {}
      ~raii() { delete[] a; delete[] b; delete[] c; }
    } buffer(s * asize + aspace - 1, s * bsize + aspace - 1, csize);
    T *const a = LIBXSMM_ALIGN(buffer.a, LIBXSMM_ALIGNED_MAX);
    T *const b = LIBXSMM_ALIGN(buffer.b, LIBXSMM_ALIGNED_MAX);
    T * /*const*/ c = buffer.c; // no alignment, but thread-local array will be aligned

#if defined(_OPENMP)
#   pragma omp parallel for CP2K_SCHEDULE
#endif
    for (int i = 0; i < s; ++i) {
      init<42>(a + i * asize, m, k, i);
      init<24>(b + i * bsize, k, n, i);
    }

#if defined(LIBXSMM_OFFLOAD)
#   pragma offload target(mic) in(a: length(s * asize)) in(b: length(s * bsize)) out(c: length(csize))
#endif
    {
#if defined(MKL_ENABLE_AVX512_MIC) && (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL))
      mkl_enable_instructions(MKL_ENABLE_AVX512_MIC);
#endif
      fprintf(stdout, "m=%i n=%i k=%i ldc=%i (%s) size=%i batch=%i memory=%.f MB\n\n",
        m, n, k, ldc, 0 != (LIBXSMM_ROW_MAJOR) ? "row-major" : "column-major", s, u,
        1.0 * (s * (asize + bsize) * sizeof(T)) / (1 << 20));

#if defined(CP2K_CHECK)
      LIBXSMM_TARGET(mic) struct LIBXSMM_TARGET(mic) raii { // avoid std::vector (first-touch init. causes NUMA issue)
        T *expect;
        explicit raii(int size): expect(new T[size]) {}
        ~raii() { delete[] expect; }
      } buffer(csize);
      T *const expect = buffer.expect;
#endif

      { // LAPACK/BLAS3 (fallback/reference)
        fprintf(stdout, "LAPACK/BLAS...\n");
        std::fill_n(expect, csize, 0);
#if defined(_OPENMP)
        double duration = -omp_get_wtime();
#       pragma omp parallel for CP2K_SCHEDULE
#endif
        for (int i = 0; i < s; i += u) {
          LIBXSMM_ALIGNED(T tmp[CP2K_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
          for (int j = 0; j < (CP2K_MAX_SIZE); ++j) tmp[j] = 0; // clear
          for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            libxsmm_blasmm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
          }
          add(expect, tmp, m, n, ldc); // atomic
        }
#if defined(_OPENMP)
        duration += omp_get_wtime();
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.1f ms\n", 1000.0 * duration);
#endif
      }

      { // inline an optimized implementation
        fprintf(stdout, "Inlined...\n");
        std::fill_n(c, csize, 0);
#if defined(_OPENMP)
        double duration = -omp_get_wtime();
#       pragma omp parallel for CP2K_SCHEDULE
#endif
        for (int i = 0; i < s; i += u) {
          LIBXSMM_ALIGNED(T tmp[CP2K_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
          for (int j = 0; j < (CP2K_MAX_SIZE); ++j) tmp[j] = 0; // clear
          for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            libxsmm_imm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
          }
          add(c, tmp, m, n, ldc); // atomic
        }
#if defined(_OPENMP)
        duration += omp_get_wtime();
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
        double duration = -omp_get_wtime();
#       pragma omp parallel for CP2K_SCHEDULE
#endif
        for (int i = 0; i < s; i += u) {
          LIBXSMM_ALIGNED(T tmp[CP2K_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
          for (int j = 0; j < (CP2K_MAX_SIZE); ++j) tmp[j] = 0; // clear
          for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            libxsmm_mm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
          }
          add(c, tmp, m, n, ldc); // atomic
        }
#if defined(_OPENMP)
        duration += omp_get_wtime();
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
        double duration = -omp_get_wtime();
#       pragma omp parallel for CP2K_SCHEDULE
#endif
        for (int i = 0; i < s; i += u) {
          LIBXSMM_ALIGNED(T tmp[CP2K_MAX_SIZE], LIBXSMM_ALIGNED_MAX);
          for (int j = 0; j < (CP2K_MAX_SIZE); ++j) tmp[j] = 0; // clear
          for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            xmm(&a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
          }
          add(c, tmp, m, n, ldc); // atomic
        }
#if defined(_OPENMP)
        duration += omp_get_wtime();
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
