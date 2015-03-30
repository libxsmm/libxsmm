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
#include <vector>
#include <cmath>

#if defined(_OPENMP)
# include <omp.h>
#endif

#if defined(LIBXSMM_OFFLOAD)
# pragma offload_attribute(pop)
#endif

#if (0 < (LIBXSMM_ALIGNED_STORES))
# define SMM_ALIGNMENT LIBXSMM_ALIGNED_STORES
#else
# define SMM_ALIGNMENT LIBXSMM_ALIGNMENT
#endif

// make sure that stacksize is covering the problem size
#define SMM_MAX_PROBLEM_SIZE (1 * LIBXSMM_MAX_MNK)
// ensures sufficient parallel slack
#define SMM_MIN_NPARALLEL 210
// ensures amortized atomic overhead
#define SMM_MIN_NLOCAL 150
#define SMM_MAX_NLOCAL 250
// OpenMP schedule policy (and chunk size)
#define SMM_SCHEDULE static,1
// enable result validation
#define SMM_CHECK


template<typename T>
LIBXSMM_TARGET(mic) void nrand(T& a)
{
  static const double scale = 1.0 / RAND_MAX;
  a = static_cast<T>(scale * (2 * std::rand() - RAND_MAX));
}


template<typename T>
LIBXSMM_TARGET(mic) void add(T *LIBXSMM_RESTRICT dst, const T *LIBXSMM_RESTRICT c, int m, int n, int ldc)
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


template<typename T>
LIBXSMM_TARGET(mic) double max_diff(const T *LIBXSMM_RESTRICT result, const T *LIBXSMM_RESTRICT expect, int m, int n, int ldc)
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
    const int default_psize = (SMM_MIN_NPARALLEL) * (SMM_MIN_NLOCAL), default_batch = SMM_MAX_NLOCAL;
    const int m = 1 < argc ? std::atoi(argv[1]) : 23;
    const int s = 2 < argc ? (0 < std::atoi(argv[2]) ? std::atoi(argv[2]) : ('+' == *argv[2]
      ? (default_psize << std::strlen(argv[2])) : ('-' == *argv[2]
      ? (default_psize >> std::strlen(argv[2])) : default_psize))) : default_psize;
    const int t = 3 < argc ? (0 < std::atoi(argv[3]) ? std::atoi(argv[3]) : ('+' == *argv[3]
      ? (default_batch << std::strlen(argv[3])) : ('-' == *argv[3]
      ? (default_batch >> std::strlen(argv[3])) : -1))) : -1;
    const int n = 4 < argc ? std::atoi(argv[4]) : m;
    const int k = 5 < argc ? std::atoi(argv[5]) : m;

    if ((SMM_MAX_PROBLEM_SIZE) < (m * n * k)) {
      throw std::runtime_error("The size M x N x K is exceeding SMM_MAX_PROBLEM_SIZE!");
    }

#if (0 != LIBXSMM_ROW_MAJOR)
# if (0 < LIBXSMM_ALIGNED_STORES)
    const int ldc = LIBXSMM_ALIGN_VALUE(int, T, n, LIBXSMM_ALIGNED_STORES);
# else
    const int ldc = n;
# endif
    const int csize = m * ldc;
#else
# if (0 < LIBXSMM_ALIGNED_STORES)
    const int ldc = LIBXSMM_ALIGN_VALUE(int, T, m, LIBXSMM_ALIGNED_STORES);
# else
    const int ldc = m;
# endif
    const int csize = n * ldc;
#endif

    const int asize = m * k, bsize = k * n;
    std::vector<T> va(s * asize), vb(s * bsize), vc(csize);
    std::for_each(va.begin(), va.end(), nrand<T>);
    std::for_each(vb.begin(), vb.end(), nrand<T>);
    const T *const a = &va[0], *const b = &vb[0];
    T * /*const*/ c = &vc[0];

#if defined(LIBXSMM_OFFLOAD)
#   pragma offload target(mic) in(a: length(s * asize)) in(b: length(s * bsize)) out(c: length(csize))
#endif
    {
      const double mbytes = 1.0 * s * (asize + bsize) * sizeof(T) / (1024 * 1024);
#if defined(_OPENMP)
      const double nbytes = 1.0 * s * (csize) * sizeof(T) / (1024 * 1024);
      const double gflops = 2.0 * s * m * n * k * 1E-9;
      const int u = 0 < t ? t : std::max(SMM_MIN_NLOCAL, std::min(SMM_MAX_NLOCAL, s / std::max(SMM_MIN_NPARALLEL, s / default_batch)));
#else
      const int u = t;
#endif
#if defined(SMM_CHECK)
      std::vector<T> expect(csize);
#endif
      fprintf(stdout, "m=%i n=%i k=%i ldc=%i size=%i batch=%i memory=%.1f MB\n\n", m, n, k, ldc, s, u, mbytes);

      { // LAPACK/BLAS3 (fallback)
        fprintf(stdout, "LAPACK/BLAS...\n");
        std::fill_n(c, csize, 0);
#if defined(_OPENMP)
        const double start = omp_get_wtime();
#       pragma omp parallel for schedule(SMM_SCHEDULE)
#endif
        for (int i = 0; i < s; i += u) {
          LIBXSMM_ALIGNED(T tmp[SMM_MAX_PROBLEM_SIZE], SMM_ALIGNMENT);
          for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
          for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            libxsmm_blasmm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
          }
          add(c, tmp, m, n, ldc); // atomic
        }
#if defined(_OPENMP)
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", (mbytes + nbytes) * 1E-3 / duration);
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(SMM_CHECK)
        std::copy(c, c + csize, expect.begin());
#endif
      }

      { // inline an optimized implementation
        fprintf(stdout, "Inlined...\n");
        std::fill_n(c, csize, 0);
#if defined(_OPENMP)
        const double start = omp_get_wtime();
#       pragma omp parallel for schedule(SMM_SCHEDULE)
#endif
        for (int i = 0; i < s; i += u) {
          LIBXSMM_ALIGNED(T tmp[SMM_MAX_PROBLEM_SIZE], SMM_ALIGNMENT);
          for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
          for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            libxsmm_imm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
          }
          add(c, tmp, m, n, ldc); // atomic
        }
#if defined(_OPENMP)
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", (mbytes + nbytes) * 1E-3 / duration);
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(SMM_CHECK)
        fprintf(stdout, "\tdiff=%f\n", max_diff(c, &expect[0], m, n, ldc));
#endif
      }

      { // auto-dispatched
        fprintf(stdout, "Dispatched...\n");
        libxsmm_mm(1, 1, 1, &a[0], &b[0], &c[0]); // warmup/workaround
        std::fill_n(c, csize, 0);
#if defined(_OPENMP)
        const double start = omp_get_wtime();
#       pragma omp parallel for schedule(SMM_SCHEDULE)
#endif
        for (int i = 0; i < s; i += u) {
          LIBXSMM_ALIGNED(T tmp[SMM_MAX_PROBLEM_SIZE], SMM_ALIGNMENT);
          for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
          for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            libxsmm_mm(m, n, k, &a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
          }
          add(c, tmp, m, n, ldc); // atomic
        }
#if defined(_OPENMP)
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", (mbytes + nbytes) * 1E-3 / duration);
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(SMM_CHECK)
        fprintf(stdout, "\tdiff=%f\n", max_diff(c, &expect[0], m, n, ldc));
#endif
      }

      const libxsmm_mm_dispatch<T> xmm(m, n, k);
      if (xmm) { // specialized routine
        fprintf(stdout, "Specialized...\n");
        std::fill_n(c, csize, 0);
#if defined(_OPENMP)
        const double start = omp_get_wtime();
#       pragma omp parallel for schedule(SMM_SCHEDULE)
#endif
        for (int i = 0; i < s; i += u) {
          LIBXSMM_ALIGNED(T tmp[SMM_MAX_PROBLEM_SIZE], SMM_ALIGNMENT);
          for (int j = 0; j < csize; ++j) tmp[j] = 0; // clear
          for (int j = 0; j < LIBXSMM_MIN(u, s - i); ++j) {
            xmm(&a[0] + (i + j) * asize, &b[0] + (i + j) * bsize, tmp);
          }
          add(c, tmp, m, n, ldc); // atomic
        }
#if defined(_OPENMP)
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", (mbytes + nbytes) * 1E-3 / duration);
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
#if defined(SMM_CHECK)
        fprintf(stdout, "\tdiff=%f\n", max_diff(c, &expect[0], m, n, ldc));
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
