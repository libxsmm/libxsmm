/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
#if defined(__MKL)
# include <mkl_service.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif

#if !defined(CHECK) && \
  (!defined(__BLAS) || (0 != __BLAS)) && /* BLAS evailable */ \
  (LIBXSMM_EQUAL(REAL_TYPE, float) || LIBXSMM_EQUAL(REAL_TYPE, double))
# define CHECK
#endif


LIBXSMM_GEMM_SYMBOL_DECL(LIBXSMM_GEMM_CONST, REAL_TYPE);


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void init(libxsmm_blasint seed, REAL_TYPE *LIBXSMM_RESTRICT dst,
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
  const libxsmm_blasint m = LIBXSMM_DEFAULT(512, 1 < argc ? atoi(argv[1]) : 0);
  const libxsmm_blasint k = LIBXSMM_DEFAULT(m, 3 < argc ? atoi(argv[3]) : 0);
  const libxsmm_blasint n = LIBXSMM_DEFAULT(k, 2 < argc ? atoi(argv[2]) : 0);
  const libxsmm_blasint lda = LIBXSMM_DEFAULT(m, 4 < argc ? atoi(argv[4]) : 0);
  const libxsmm_blasint ldb = LIBXSMM_DEFAULT(k, 5 < argc ? atoi(argv[5]) : 0);
  const libxsmm_blasint ldc = LIBXSMM_DEFAULT(m, 6 < argc ? atoi(argv[6]) : 0);
  const REAL_TYPE alpha = (REAL_TYPE)(7 < argc ? atof(argv[7]) : 1.0);
  const REAL_TYPE beta  = (REAL_TYPE)(8 < argc ? atof(argv[8]) : 1.0);
  const int nrepeat = LIBXSMM_DEFAULT(
    LIBXSMM_MAX(13 / LIBXSMM_MAX(1, libxsmm_cbrt_u64(1ULL * m * n * k) >> 10), 3),
    9 < argc ? atoi(argv[9]) : 0);
  const double gflops = 2.0 * m * n * k * 1E-9;
  const char transa = 'N', transb = 'N';
  int result = EXIT_SUCCESS;
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    const char *const env_tasks = getenv("TASKS");
    const int tasks = (0 == env_tasks || 0 == *env_tasks) ? 0/*default*/ : atoi(env_tasks);
    REAL_TYPE *const a = (REAL_TYPE*)libxsmm_malloc((size_t)(lda * k * sizeof(REAL_TYPE)));
    REAL_TYPE *const b = (REAL_TYPE*)libxsmm_malloc((size_t)(ldb * n * sizeof(REAL_TYPE)));
    REAL_TYPE *const c = (REAL_TYPE*)libxsmm_malloc((size_t)(ldc * n * sizeof(REAL_TYPE)));
#if defined(CHECK)
    REAL_TYPE* d = 0;
    if (!LIBXSMM_FEQ(0, check)) {
      d = (REAL_TYPE*)libxsmm_malloc((size_t)(ldc * n * sizeof(REAL_TYPE)));
      init(0, d, m, n, ldc, 1.0);
    }
#endif
    init( 0, c, m, n, ldc, 1.0);
    init(42, a, m, k, lda, 1.0);
    init(24, b, k, n, ldb, 1.0);
#if defined(MKL_ENABLE_AVX512)
    mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
    /* warmup OpenMP (populate thread pool) */
    LIBXSMM_YGEMM_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#if defined(CHECK)
    if (0 != d) {
      LIBXSMM_GEMM_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, d, &ldc);
    }
#endif
    libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    fprintf(stdout, "\n\n");

    if (0 == tasks) { /* tiled xGEMM (with library-internal parallelization) */
      int i; double duration;
      unsigned long long start = libxsmm_timer_tick();
      for (i = 0; i < nrepeat; ++i) {
        LIBXSMM_YGEMM_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
      }
    }
    else { /* tiled xGEMM (with external parallelization) */
      int i; double duration;
      unsigned long long start = libxsmm_timer_tick();
      for (i = 0; i < nrepeat; ++i) {
#if defined(_OPENMP)
#       pragma omp parallel
#       pragma omp single nowait
#endif
        LIBXSMM_YGEMM_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
      }
    }
#if defined(CHECK)
    if (0 != d) { /* validate result against LAPACK/BLAS xGEMM */
      libxsmm_matdiff_info diff;
      int i; double duration;
      unsigned long long start = libxsmm_timer_tick();
      for (i = 0; i < nrepeat; ++i) {
        LIBXSMM_GEMM_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, d, &ldc);
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
      }
      if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(REAL_TYPE), m, n, d, c, &ldc, &ldc, &diff)) {
        fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
        if (check < 100.0 * diff.normf_rel) {
          fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
          result = EXIT_FAILURE;
        }
      }
      libxsmm_free(d);
    }
#endif
    libxsmm_free(c);
    libxsmm_free(a);
    libxsmm_free(b);
  }
  fprintf(stdout, "Finished\n");

  return result;
}
