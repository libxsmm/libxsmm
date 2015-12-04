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
#include <libxsmm_timer.h>

#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
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

#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(pop)
#endif

#define MAX_SIZE (80 * 80)


template<int Seed>
struct LIBXSMM_RETARGETABLE init {
  template<typename T> init(T *LIBXSMM_RESTRICT dst, int nrows, int ncols, int n = 0, int ld = 0) {
    const int ldx = 0 == ld ? ncols : ld;
    const int minval = n + Seed, addval = (nrows - 1) * ldx + (ncols - 1);
    const int maxval = std::max(std::abs(minval), addval);
    const double norm = 0 != maxval ? (1.0 / maxval) : 1.0;
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        const double value = static_cast<double>(i * ldx + j + minval);
        dst[i*ldx+j] = static_cast<T>(norm * (value - 0.5 * addval));
      }
    }
  }
};


int main(int argc, char* argv[])
{
  try {
    typedef double T;
    const int m = 1 < argc ? std::atoi(argv[1]) : 23;
    const int n = 2 < argc ? std::atoi(argv[2]) : m;
    const int k = 3 < argc ? std::atoi(argv[3]) : m;

    const int csize = m * n;
    if ((MAX_SIZE) < csize) {
      throw std::runtime_error("The size M x N is exceeding MAX_SIZE!");
    }

    const int asize = m * k, bsize = k * n, aspace = LIBXSMM_ALIGNMENT / sizeof(T);
    const int s = (2ULL << 30) / ((asize + bsize + csize) * sizeof(T)); // 2 GByte
    const size_t bwsize_batched = (asize/*load*/ + bsize/*load*/ + 2 * csize/*RFO*/) * sizeof(T); // batched
    const size_t bwsize = (asize/*load*/ + bsize/*load*/) * sizeof(T); // streamed, skipping C since it is just in cache
    const double gflops = 2.0 * s * m * n * k * 1E-9;

    struct raii { // avoid std::vector (first-touch init. causes NUMA issue)
      T *a, *b, *c;
      raii(int asize, int bsize, int csize): a(new T[asize]), b(new T[bsize]), c(new T[csize]) {}
      ~raii() { delete[] a; delete[] b; delete[] c; }
    } buffer(s * asize + aspace - 1, s * bsize + aspace - 1, s * csize + aspace - 1);
    T *const a = LIBXSMM_ALIGN2(buffer.a, LIBXSMM_ALIGNMENT);
    T *const b = LIBXSMM_ALIGN2(buffer.b, LIBXSMM_ALIGNMENT);
    T *c = LIBXSMM_ALIGN2(buffer.c, LIBXSMM_ALIGNMENT);

#if defined(_OPENMP)
#   pragma omp parallel for
#endif
    for (int i = 0; i < s; ++i) {
      init<42>(a + i * asize, m, k, i);
      init<24>(b + i * bsize, k, n, i);
      init<22>(c + i * csize, m, n, i);
    }

#if defined(LIBXSMM_OFFLOAD_BUILD)
#   pragma offload target(LIBXSMM_OFFLOAD_TARGET) in(a: length(s * asize)) in(b: length(s * bsize)) inout(c: length(s * csize))
#endif
    {
#if defined(MKL_ENABLE_AVX512_MIC)
      mkl_enable_instructions(MKL_ENABLE_AVX512_MIC);
#endif
      // initialize LIBXSMM
      libxsmm_init();

      fprintf(stdout, "m=%i n=%i k=%i (%s, %s) size=%i memory=%.f MB\n\n", m, n, k,
        0 != LIBXSMM_ROW_MAJOR ? "row-major" : "column-major", 8 == sizeof(T) ? "DP" : "SP",
        s, 1.0 * (s * (asize + bsize + csize) * sizeof(T)) / (1 << 20));

      { // batched
        fprintf(stdout, "Batched (A,B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#       pragma omp parallel for
#endif
        for (int i = 0; i < s; ++i) {
          LIBXSMM_INLINE_GEMM(LIBXSMM_FLAGS, m, n, k,
            LIBXSMM_ALPHA, a + i * asize, m, b + i * bsize, k,
            LIBXSMM_BETA, c + i * csize, m);
        }
        const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize_batched / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }

      { // streaming
        fprintf(stdout, "Streamed (A,B)...\n");
        const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#       pragma omp parallel for
#endif
        for (int i = 0; i < s; ++i) {
          // make sure that stacksize is covering the problem size
          T buffer[MAX_SIZE]; // LIBXSMM_ALIGNED does not apply to non-static local stack variables
          T *const tmp = LIBXSMM_ALIGN2(buffer, LIBXSMM_ALIGNMENT);
          // do nothing else with tmp; just a benchmark
          LIBXSMM_INLINE_GEMM(LIBXSMM_FLAGS, m, n, k,
            LIBXSMM_ALPHA, a + i * asize, m, b + i * bsize, k,
            LIBXSMM_BETA, tmp, m);
        }
        const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }

      { // cached
        fprintf(stdout, "Cached...\n");
        const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#       pragma omp parallel for
#endif
        for (int i = 0; i < s; ++i) {
          // make sure that stacksize is covering the problem size
          T buffer[MAX_SIZE]; // LIBXSMM_ALIGNED does not apply to non-static local stack variables
          T *const tmp = LIBXSMM_ALIGN2(buffer, LIBXSMM_ALIGNMENT);
          // do nothing else with tmp; just a benchmark
          LIBXSMM_INLINE_GEMM(LIBXSMM_FLAGS, m, n, k,
            LIBXSMM_ALPHA, a, m, b, k,
            LIBXSMM_BETA, tmp, m);
        }
        const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }

      // finalize LIBXSMM
      libxsmm_finalize();
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
