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
/* Christopher Dahnken (Intel Corp.), Hans Pabst (Intel Corp.),
 * Alfio Lazzaro (CRAY Inc.), and Gilles Fourestey (CSCS)
******************************************************************************/
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>

#include <libxsmm.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

// make sure that stacksize is covering the problem size
#define SMM_MAX_PROBLEM_SIZE (5 * LIBXSMM_MAX_MNK)

#if (0 < (LIBXSMM_ALIGNED_STORES))
# define SMM_ALIGNMENT LIBXSMM_ALIGNED_STORES
#else
# define SMM_ALIGNMENT LIBXSMM_ALIGNMENT
#endif

#define SMM_CHECK


template<typename T> void nrand(T& a)
{
  static const double scale = 1.0 / RAND_MAX;
  a = static_cast<T>(scale * (2 * std::rand() - RAND_MAX));
}


template<typename T> void add(T *LIBXSMM_RESTRICT dst, const T *LIBXSMM_RESTRICT c, int m, int n, int ldc)
{
#if (0 < LIBXSMM_ALIGNED_STORES)
  LIBXSMM_ASSUME_ALIGNED(c, SMM_ALIGNMENT);
#endif
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
#if (0 != LIBXSMM_ROW_MAJOR)
      const T value = c[i*ldc+j];
#else
      const T value = c[j*ldc+i];
#endif
#if defined(_OPENMP)
#     pragma omp atomic
#endif
#if (0 != LIBXSMM_ROW_MAJOR)
      dst[i*n+j] += value;
#else
      dst[j*m+i] += value;
#endif
    }
  }
}


template<typename T> double max_diff(const T *LIBXSMM_RESTRICT result, const T *LIBXSMM_RESTRICT expect, int m, int n, int ldc)
{
  double diff = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
#if (0 != LIBXSMM_ROW_MAJOR)
      const int k = i * ldc + j;
#else
      const int k = j * ldc + i;
#endif
      diff = std::max(diff, std::abs(static_cast<double>(result[k]) - static_cast<double>(expect[k])));
    }
  }
  return diff;
}


int main(int argc, char* argv[])
{
  try {
    typedef double T;
    const int default_psize = 500000, default_batch = 100;
    const int m = 1 < argc ? std::atoi(argv[1]) : 23;
    const int s = 2 < argc ? (0 < std::atoi(argv[2]) ? std::atoi(argv[2]) : ('+' == *argv[2]
      ? (default_psize << std::strlen(argv[2]))
      : (default_psize >> std::strlen(argv[2])))) : default_psize;
    const int t = 3 < argc ? (0 < std::atoi(argv[3]) ? std::atoi(argv[3]) : ('+' == *argv[3]
      ? (default_batch << std::strlen(argv[3]))
      : (default_batch >> std::strlen(argv[3])))) : default_batch;
    const int n = 4 < argc ? std::atoi(argv[4]) : m;
    const int k = 5 < argc ? std::atoi(argv[5]) : m;

#if (0 != LIBXSMM_ALIGNED_STORES)
# if (0 != LIBXSMM_ROW_MAJOR)
    const int ldc = LIBXSMM_ALIGN_VALUE(int, T, n, LIBXSMM_ALIGNED_STORES);
    const int csize = m * ldc + (LIBXSMM_ALIGNED_STORES) / sizeof(T) - 1;
# else
    const int ldc = LIBXSMM_ALIGN_VALUE(int, T, m, LIBXSMM_ALIGNED_STORES);
    const int csize = n * ldc + (LIBXSMM_ALIGNED_STORES) / sizeof(T) - 1;
# endif
#else
# if (0 != LIBXSMM_ROW_MAJOR)
    const int ldc = n, csize = m * ldc;
# else
    const int ldc = m, csize = n * ldc;
# endif
#endif

#if defined(_OPENMP)
    const double gflops = (2ULL * s * m * n * k) * 1E-9;
#endif
    const int asize = m * k, bsize = k * n;
    std::vector<T> a(s * asize), b(s * bsize), result(csize);
#if defined(SMM_CHECK)
    std::vector<T> expect(csize);
#endif
    std::for_each(a.begin(), a.end(), nrand<T>);
    std::for_each(b.begin(), b.end(), nrand<T>);
    fprintf(stdout, "psize=%i batch=%i m=%i n=%i k=%i ldc=%i\n", s, t, m, n, k, ldc);

    { // LAPACK/BLAS3 (fallback)
      fprintf(stdout, "LAPACK/BLAS...\n");
      std::fill(result.begin(), result.end(), 0);
#if defined(_OPENMP)
      const double start = omp_get_wtime();
#endif
#     pragma omp parallel for
      for (int i = 0; i < s; i += t) {
        LIBXSMM_ALIGNED(T tmp[SMM_MAX_PROBLEM_SIZE], SMM_ALIGNMENT);
        for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
        for (int j = 0; j < LIBXSMM_MIN(t, s - i); ++j) {
          libxsmm_blasmm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
        }
        add(&result[0], tmp, m, n, ldc); // atomic
      }
#if defined(_OPENMP)
      const double duration = omp_get_wtime() - start;
      if (0 < duration) {
        fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
      }
      fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(SMM_CHECK)
      std::copy(result.begin(), result.end(), expect.begin());
#endif
    }

    { // auto-dispatched
      fprintf(stdout, "Dispatched...\n");
      std::fill(result.begin(), result.end(), 0);
#if defined(_OPENMP)
      const double start = omp_get_wtime();
#endif
#     pragma omp parallel for
      for (int i = 0; i < s; i += t) {
        LIBXSMM_ALIGNED(T tmp[SMM_MAX_PROBLEM_SIZE], SMM_ALIGNMENT);
        for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
        for (int j = 0; j < LIBXSMM_MIN(t, s - i); ++j) {
          libxsmm_mm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
        }
        add(&result[0], tmp, m, n, ldc); // atomic
      }
#if defined(_OPENMP)
      const double duration = omp_get_wtime() - start;
      if (0 < duration) {
        fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
      }
      fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(SMM_CHECK)
      fprintf(stdout, "\tdiff=%f\n", max_diff(&result[0], &expect[0], m, n, ldc));
#endif
    }

    { // inline an optimized implementation
      fprintf(stdout, "Inlined...\n");
      std::fill(result.begin(), result.end(), 0);
#if defined(_OPENMP)
      const double start = omp_get_wtime();
#endif
#     pragma omp parallel for
      for (int i = 0; i < s; i += t) {
        LIBXSMM_ALIGNED(T tmp[SMM_MAX_PROBLEM_SIZE], SMM_ALIGNMENT);
        for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
        for (int j = 0; j < LIBXSMM_MIN(t, s - i); ++j) {
          libxsmm_imm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
        }
        add(&result[0], tmp, m, n, ldc); // atomic
      }
#if defined(_OPENMP)
      const double duration = omp_get_wtime() - start;
      if (0 < duration) {
        fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
      }
      fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(SMM_CHECK)
      fprintf(stdout, "\tdiff=%f\n", max_diff(&result[0], &expect[0], m, n, ldc));
#endif
    }

    const libxsmm_mm_dispatch<T> xmm(m, n, k);
    if (xmm) { // specialized routine
      fprintf(stdout, "Specialized...\n");
      std::fill(result.begin(), result.end(), 0);
#if defined(_OPENMP)
      const double start = omp_get_wtime();
#endif
#     pragma omp parallel for
      for (int i = 0; i < s; i += t) {
        LIBXSMM_ALIGNED(T tmp[SMM_MAX_PROBLEM_SIZE], SMM_ALIGNMENT);
        for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
        for (int j = 0; j < LIBXSMM_MIN(t, s - i); ++j) {
          xmm(&a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
        }
        add(&result[0], tmp, m, n, ldc); // atomic
      }
#if defined(_OPENMP)
      const double duration = omp_get_wtime() - start;
      if (0 < duration) {
        fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
      }
      fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(SMM_CHECK)
      fprintf(stdout, "\tdiff=%f\n", max_diff(&result[0], &expect[0], m, n, ldc));
#endif
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
