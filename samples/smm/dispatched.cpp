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


template<int Seed>
struct LIBXSMM_TARGET(mic) init {
  template<typename T> init(T *LIBXSMM_RESTRICT dst, int nrows, int ncols, int n = 0, int ld = 0) {
    const int ldx = 0 == ld ? ncols : ld;
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        // initialize similar to CP2K's (libsmm_acc) benchmark driver
        dst[i*ldx+j] = static_cast<T>(i * ldx + j + n + Seed);
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

    const int asize = m * k, bsize = k * n, aspace = (LIBXSMM_ALIGNED_MAX) / sizeof(T);
    const int ldc = LIBXSMM_ALIGN_STORES(LIBXSMM_LD(m, n), sizeof(T)), csize = LIBXSMM_LD(n, m) * ldc;
    const int s = (3ULL << 30) / ((asize + bsize + csize) * sizeof(T)); // 3 GByte
    const double gbytes = 1.0 * s * (asize + bsize + csize) * sizeof(T) / (1 << 30);
#if defined(_OPENMP)
    const size_t bwsize = (asize/*load*/ + bsize/*load*/ + csize * 2/*load and store*/) * sizeof(T); // cached
    const double gflops = 2.0 * s * m * n * k * 1E-9;
#endif

    struct raii { // avoid std::vector (first-touch init. causes NUMA issue)
      T *a, *b;
      raii(int s, int asize, int bsize, int aspace)
        : a(new T[s*asize+aspace-1]), b(new T[s*bsize+aspace-1])
      {}
      ~raii() { delete[] a; delete[] b; }
    } buffer(s, asize, bsize, aspace);
    T *const a = LIBXSMM_ALIGN(buffer.a, LIBXSMM_ALIGNED_MAX);
    T *const b = LIBXSMM_ALIGN(buffer.b, LIBXSMM_ALIGNED_MAX);

#if defined(_OPENMP)
#   pragma omp parallel for
#endif
    for (int i = 0; i < s; ++i) {
      init<42>(a + i * asize, m, k, i);
      init<24>(b + i * bsize, k, n, i);
    }

#if defined(LIBXSMM_OFFLOAD)
#   pragma offload target(mic) in(a: length(s * asize)) in(b: length(s * bsize))
#endif
    {
#if defined(USE_MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
      mkl_enable_instructions(MKL_ENABLE_AVX512_MIC);
#endif
      fprintf(stdout, "m=%i n=%i k=%i ldc=%i (%s) size=%i memory=%.f MB\n\n",
        m, n, k, ldc, 0 != (LIBXSMM_ROW_MAJOR) ? "row-major" : "column-major", s, 1024 * gbytes);

      { // streaming
        fprintf(stdout, "Streamed...\n");
#if defined(_OPENMP)
        double start = 0;
#       pragma omp parallel
        {
#         pragma omp master
          start = omp_get_wtime();
#         pragma omp for
#endif
          for (int i = 0; i < s; ++i) {
            // make sure that stacksize is covering the problem size; tmp is zero-initialized by lang. rules
            LIBXSMM_ALIGNED(T tmp[LIBXSMM_MAX_SIZE/*max. problemsize*/], LIBXSMM_ALIGNED_MAX);
            libxsmm_mm(m, n, k, a + i * asize, b + i * bsize, tmp);
          }
#if defined(_OPENMP)
        }
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
#endif
      }

      { // cached
        fprintf(stdout, "Cached...\n");
#if defined(_OPENMP)
        double start = 0;
#       pragma omp parallel
        {
#         pragma omp master
          start = omp_get_wtime();
# if defined(__MIC__)
#         pragma omp for schedule(dynamic)
# else
#         pragma omp for
# endif
#endif
          for (int i = 0; i < s; ++i) {
            // make sure that stacksize is covering the problem size; tmp is zero-initialized by lang. rules
            LIBXSMM_ALIGNED(T tmp[LIBXSMM_MAX_SIZE/*max. problemsize*/], LIBXSMM_ALIGNED_MAX);
            // do nothing else with tmp; just a benchmark
            libxsmm_mm(m, n, k, a, b, tmp);
          }
#if defined(_OPENMP)
        }
        const double duration = omp_get_wtime() - start;
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
        }
        fprintf(stdout, "\tduration: %.1f s\n", duration);
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
