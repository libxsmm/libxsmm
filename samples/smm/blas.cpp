/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
# include <mkl.h>
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

#define MAX_SIZE (80 * 80)


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void init(int seed, REAL_TYPE *LIBXSMM_RESTRICT dst,
  libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
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


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  try {
    typedef REAL_TYPE T;
    const int m = 1 < argc ? std::atoi(argv[1]) : 23;
    const int n = 2 < argc ? std::atoi(argv[2]) : m;
    const int k = 3 < argc ? std::atoi(argv[3]) : m;
    const unsigned long size = LIBXSMM_DEFAULT(2048/*default: 2 GByte*/,
      4 < argc ? std::strtoul(argv[4], 0, 10) : 0) << 20;

    const int asize = m * k, bsize = k * n, csize = m * n, aspace = LIBXSMM_ALIGNMENT / sizeof(T);
    const int s = LIBXSMM_MAX(size / ((asize + bsize + csize) * sizeof(T)), 1);
    const size_t bwsize_batched = (asize/*load*/ + bsize/*load*/ + 2 * csize/*RFO*/) * sizeof(T); // batched
    const size_t bwsize = (asize/*load*/ + bsize/*load*/) * sizeof(T); // streamed, skipping C since it is just in cache
    const double gflops = 2.0 * s * m * n * k * 1E-9, scale = 1.0;

    struct raii { // avoid std::vector (first-touch init. causes NUMA issue)
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
      T *a, *b, *c, *d;
      raii(int asize_, int bsize_, int csize_): a(new T[asize_]), b(new T[bsize_]), c(new T[csize_]), d(new T[csize_]) {}
      ~raii() { delete[] a; delete[] b; delete[] c; delete[] d; }
#else
      T *a, *b, *c;
      raii(int asize, int bsize, int csize): a(new T[asize]), b(new T[bsize]), c(new T[csize]) {}
      ~raii() { delete[] a; delete[] b; delete[] c; }
#endif
    } buffer(s * asize + aspace - 1, s * bsize + aspace - 1, s * csize + aspace - 1);
    T *const a = LIBXSMM_ALIGN2(buffer.a, LIBXSMM_ALIGNMENT);
    T *const b = LIBXSMM_ALIGN2(buffer.b, LIBXSMM_ALIGNMENT);
    T *c = LIBXSMM_ALIGN2(buffer.c, LIBXSMM_ALIGNMENT);
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
    T *d = LIBXSMM_ALIGN2(buffer.c, LIBXSMM_ALIGNMENT);
#endif

#if defined(_OPENMP)
#   pragma omp parallel for
#endif
    for (int i = 0; i < s; ++i) {
      init(42 + i, a + i * asize, m, k, m, scale);
      init(24 + i, b + i * bsize, k, n, k, scale);
      init(22 + i, c + i * csize, m, n, m, scale);
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
      init(22 + i, d + i * csize, m, n, m, scale);
#endif
    }

#if defined(LIBXSMM_OFFLOAD_TARGET)
# if defined(__MKL) && (2 == __MKL)
#   pragma offload target(LIBXSMM_OFFLOAD_TARGET) in(a: length(s * asize)) in(b: length(s * bsize)) inout(c: length(s * csize)) inout(d: length(s * csize))
# else
#   pragma offload target(LIBXSMM_OFFLOAD_TARGET) in(a: length(s * asize)) in(b: length(s * bsize)) inout(c: length(s * csize))
# endif
#endif
    {
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      // initialize LIBXSMM
      libxsmm_init();

      fprintf(stdout, "m=%i n=%i k=%i size=%i memory=%.1f MB (%s)\n\n", m, n, k, s,
        1.0 * (s * (asize + bsize + csize) * sizeof(T)) / (1 << 20), 8 == sizeof(T) ? "DP" : "SP");

      { // LAPACK/BLAS3 (warmup BLAS Library)
#if defined(_OPENMP)
#       pragma omp parallel for
#endif
        for (int i = 0; i < s; ++i) {
          // alternatively libxsmm_blas_gemm can be called instead of relying on a macro
          LIBXSMM_BLAS_GEMM(LIBXSMM_FLAGS, m, n, k,
            LIBXSMM_ALPHA, a + i * asize, LIBXSMM_LD(m, k), b + i * bsize, LIBXSMM_LD(k, n),
            LIBXSMM_BETA, c + i * csize, LIBXSMM_LD(m, n));
        }
      }

      { // batched
        fprintf(stdout, "Batched (A,B,C)...\n");
        const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#       pragma omp parallel for
#endif
        for (int i = 0; i < s; ++i) {
          // alternatively libxsmm_blas_gemm can be called instead of relying on a macro
          LIBXSMM_BLAS_GEMM(LIBXSMM_FLAGS, m, n, k,
            LIBXSMM_ALPHA, a + i * asize, LIBXSMM_LD(m, k), b + i * bsize, LIBXSMM_LD(k, n),
            LIBXSMM_BETA, c + i * csize, LIBXSMM_LD(m, n));
        }
        const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize_batched / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }

#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && defined(INTEL_MKL_VERSION) && (110300 <= (INTEL_MKL_VERSION))
      { // MKL-batched
        fprintf(stdout, "MKL-Batched (A,B,C)...\n");
        const char transa_array[] = { 0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'N' : 'T' };
        const char transb_array[] = { 0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'N' : 'T' };
        const T alpha_array[] = { LIBXSMM_ALPHA }, beta_array[] = { LIBXSMM_BETA };
        std::vector<const T*> va_array(s), vb_array(s); std::vector<T*> vc_array(s);
        const T* *const a_array = &va_array[0];
        const T* *const b_array = &vb_array[0];
        T* *const c_array = &vc_array[0];
        const int group_count = 1;
        for (int i = 0; i < s; ++i) {
          a_array[i] = a + i * asize; b_array[i] = b + i * bsize; c_array[i] = d + i * csize;
        }
        // additional warm-up
        LIBXSMM_TPREFIX(REAL_TYPE,gemm_batch)(transa_array, transb_array, &m, &n, &k,
          alpha_array, &a_array[0], &m, &b_array[0], &k,
           beta_array, &c_array[0], &m, &group_count, &s);
        // reset the destination after warm-up
        for (int i = 0; i < s; ++i) c_array[i] = d + i * csize;
        const unsigned long long start = libxsmm_timer_tick();
        LIBXSMM_TPREFIX(REAL_TYPE,gemm_batch)(transa_array, transb_array, &m, &n, &k,
          alpha_array, &a_array[0], &m, &b_array[0], &k,
           beta_array, &c_array[0], &m, &group_count, &s);
        const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize_batched / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        double d2 = 0;
        for (int h = 0; h < s; ++h) {
          const T *const x = c + h * csize, *const y = c_array[h];
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              const int index = i * n + j;
              const double d1 = static_cast<double>(x[index] - y[index]);
              d2 += d1 * d1;
            }
          }
        }
        fprintf(stdout, "\tdiff=%f\n", d2 / s);
      }
#endif

      if ((MAX_SIZE) >= csize) {
        { // streaming
          fprintf(stdout, "Streamed (A,B)...\n");
          const unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel for
#endif
          for (int i = 0; i < s; ++i) {
            // make sure that stacksize is covering the problem size
            T tls[MAX_SIZE]; // LIBXSMM_ALIGNED does not apply to non-static local stack variables
            T *const tmp = LIBXSMM_ALIGN_LDST(tls);
            // do nothing else with tmp; just a benchmark
            // alternatively libxsmm_blas_gemm can be called instead of relying on a macro
            LIBXSMM_BLAS_GEMM(LIBXSMM_FLAGS, m, n, k,
              LIBXSMM_ALPHA, a + i * asize, LIBXSMM_LD(m, k), b + i * bsize, LIBXSMM_LD(k, n),
              LIBXSMM_BETA, tmp, LIBXSMM_LD(m, n));
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
#         pragma omp parallel for
#endif
          for (int i = 0; i < s; ++i) {
            // make sure that stacksize is covering the problem size
            T tls[MAX_SIZE]; // LIBXSMM_ALIGNED does not apply to non-static local stack variables
            T *const tmp = LIBXSMM_ALIGN_LDST(tls);
            // do nothing else with tmp; just a benchmark
            // alternatively libxsmm_blas_gemm can be called instead of relying on a macro
            LIBXSMM_BLAS_GEMM(LIBXSMM_FLAGS, m, n, k,
              LIBXSMM_ALPHA, a, LIBXSMM_LD(m, k), b, LIBXSMM_LD(k, n),
              LIBXSMM_BETA, tmp, LIBXSMM_LD(m, n));
          }
          const double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
          if (0 < duration) {
            fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          }
          fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        }
      }
      else {
        fprintf(stderr, "Warning: size M x N is exceeding MAX_SIZE!\n");
      }

      // finalize LIBXSMM
      libxsmm_finalize();
      fprintf(stdout, "Finished\n");
    }
  }
  catch(const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    result = EXIT_FAILURE;
  }
  catch(...) {
    fprintf(stderr, "Error: unknown exception caught!\n");
    result = EXIT_FAILURE;
  }

  return result;
}
