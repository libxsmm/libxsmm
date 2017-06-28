/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif


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
  const libxsmm_blasint m = LIBXSMM_DEFAULT(2048, 1 < argc ? atoi(argv[1]) : 0);
  const libxsmm_blasint k = LIBXSMM_DEFAULT(m, 3 < argc ? atoi(argv[3]) : 0);
  const libxsmm_blasint n = LIBXSMM_DEFAULT(k, 2 < argc ? atoi(argv[2]) : 0);
  const libxsmm_blasint bm = LIBXSMM_DEFAULT(32, 4 < argc ? atoi(argv[4]) : 0);
  const libxsmm_blasint bk = LIBXSMM_DEFAULT(bm, 6 < argc ? atoi(argv[6]) : 0);
  const libxsmm_blasint bn = LIBXSMM_DEFAULT(bk, 5 < argc ? atoi(argv[5]) : 0);
  const libxsmm_bgemm_order order = LIBXSMM_DEFAULT(0, 7 < argc ? atoi(argv[7]) : 0);
  const int nrepeat = LIBXSMM_DEFAULT(100, 8 < argc ? atoi(argv[8]) : 0);
  const libxsmm_blasint lda = m, ldb = k, ldc = m;
  const double gflops = 2.0 * m * n * k * 1E-9;
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(&transa, &transb);
  const REAL_TYPE alpha = 1, beta = 1;
  int result = EXIT_SUCCESS;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) { /* check command line */
    printf("\nUsage: ./block_gemm [M] [N] [K] [bm] [bn] [bk] [order] [reps]\n\n");
    return result;
  }

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    REAL_TYPE *const agold = (REAL_TYPE*)libxsmm_malloc(lda * k * sizeof(REAL_TYPE));
    REAL_TYPE *const bgold = (REAL_TYPE*)libxsmm_malloc(ldb * n * sizeof(REAL_TYPE));
    REAL_TYPE *const cgold = (REAL_TYPE*)libxsmm_malloc(ldc * n * sizeof(REAL_TYPE));
    REAL_TYPE *const a = (REAL_TYPE*)libxsmm_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE *const b = (REAL_TYPE*)libxsmm_malloc(k * n * sizeof(REAL_TYPE));
    REAL_TYPE *const c = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    libxsmm_bgemm_handle* handle = 0;
    unsigned long long start;
    double duration;
#if !defined(__BLAS) || (0 != __BLAS)
    const char *const env_check = getenv("CHECK");
    const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
    handle = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      m, n, k, bm, bn, bk, &alpha, &beta, &gemm_flags, &order);
    init(42, agold, m, k, lda, 1.0);
    init(24, bgold, k, n, ldb, 1.0);
    init( 0, cgold, m, n, ldc, 1.0);
    libxsmm_bgemm_copyin_a(handle, agold, &lda, a);
    libxsmm_bgemm_copyin_b(handle, bgold, &ldb, b);
    libxsmm_bgemm_copyin_c(handle, cgold, &ldc, c);
#if defined(MKL_ENABLE_AVX512)
    mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
    /* warmup OpenMP (populate thread pool) */
    libxsmm_bgemm_omp(handle, a, b, c, 1);
#if !defined(__BLAS) || (0 != __BLAS)
    if (!LIBXSMM_FEQ(0, check)) {
      LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, agold, &lda, bgold, &ldb, &beta, cgold, &ldc);
    }
#endif
    libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    fprintf(stdout, "\n\n");

    start = libxsmm_timer_tick();
    libxsmm_bgemm_omp(handle, a, b, c, nrepeat);
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    if (0 < duration) {
      fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
    }
#if !defined(__BLAS) || (0 != __BLAS)
    if (!LIBXSMM_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
      libxsmm_matdiff_info matdiff_info;
      int i; double duration;
      start = libxsmm_timer_tick();
      for (i = 0; i < nrepeat; ++i) {
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, agold, &lda, bgold, &ldb, &beta, cgold, &ldc);
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
      }
      if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_GEMM_PRECISION(REAL_TYPE), m, n, cgold, c, 0/*ldref*/, &ldc, &matdiff_info)) {
        const char *const env_check_tolerance = getenv("CHECK_TOLERANCE");
        const double check_tolerance = LIBXSMM_ABS(0 == env_check_tolerance ? 0.000001 : atof(env_check_tolerance));
        fprintf(stderr, "\tdiff: L1max=%f, L1rel=%f and L2=%f\n", matdiff_info.norm_l1_max, matdiff_info.norm_l1_rel, matdiff_info.norm_l2);
        if (check_tolerance < matdiff_info.norm_l1_max) result = EXIT_FAILURE;
      }
    }
#endif
    libxsmm_bgemm_handle_destroy(handle);
    libxsmm_free(agold);
    libxsmm_free(bgold);
    libxsmm_free(cgold);
    libxsmm_free(a);
    libxsmm_free(b);
    libxsmm_free(c);
  }
  fprintf(stdout, "Finished\n");

  return result;
}
