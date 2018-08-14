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

#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif

#if !defined(XGEMM)
# define XGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_YGEMM_SYMBOL(ITYPE)(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif

#if !defined(CHECK) && \
  (!defined(__BLAS) || (0 != __BLAS)) && /* BLAS available */ \
  (LIBXSMM_EQUAL(ITYPE, float) || LIBXSMM_EQUAL(ITYPE, double))
# define XGEMM_GOLD(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_GEMM_SYMBOL(ITYPE)(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
# define CHECK
#endif


LIBXSMM_GEMM_SYMBOL_DECL(LIBXSMM_GEMM_CONST, ITYPE);


int main(int argc, char* argv[])
{
  LIBXSMM_GEMM_CONST libxsmm_blasint m = (1 < argc ? atoi(argv[1]) : 512);
  LIBXSMM_GEMM_CONST libxsmm_blasint k = (3 < argc ? atoi(argv[3]) : m);
  LIBXSMM_GEMM_CONST libxsmm_blasint n = (2 < argc ? atoi(argv[2]) : k), nn = n;
  LIBXSMM_GEMM_CONST OTYPE alpha = (OTYPE)(7 < argc ? atof(argv[7]) : 1.0);
  LIBXSMM_GEMM_CONST OTYPE beta  = (OTYPE)(8 < argc ? atof(argv[8]) : 1.0);
  LIBXSMM_GEMM_CONST char transa =  (9 < argc ? *argv[9]  : 'N');
  LIBXSMM_GEMM_CONST char transb = (10 < argc ? *argv[10] : 'N');
  LIBXSMM_GEMM_CONST libxsmm_blasint mm = (('N' == transa || 'n' == transa) ? m : k);
  LIBXSMM_GEMM_CONST libxsmm_blasint kk = (('N' == transb || 'n' == transb) ? k : n);
  LIBXSMM_GEMM_CONST libxsmm_blasint ka = (('N' == transa || 'n' == transa) ? k : m);
  LIBXSMM_GEMM_CONST libxsmm_blasint kb = (('N' == transb || 'n' == transb) ? n : k);
  LIBXSMM_GEMM_CONST libxsmm_blasint lda = ((4 < argc && mm < atoi(argv[4])) ? atoi(argv[4]) : mm);
  LIBXSMM_GEMM_CONST libxsmm_blasint ldb = ((5 < argc && kk < atoi(argv[5])) ? atoi(argv[5]) : kk);
  LIBXSMM_GEMM_CONST libxsmm_blasint ldc = ((6 < argc && m < atoi(argv[6])) ? atoi(argv[6]) : m);
  const int nrepeat = ((11 < argc && 0 < atoi(argv[11])) ? atoi(argv[11])
    : LIBXSMM_MAX(13 / LIBXSMM_MAX(1, (int)(libxsmm_icbrt_u64(1ULL * m * n * k) >> 10)), 3));
  const double gflops = 2.0 * m * n * k * 1E-9;
  int result = EXIT_SUCCESS;
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  if (ka == kk) {
#if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
    {
      const char *const env_tasks = getenv("TASKS");
      const int tasks = (0 == env_tasks || 0 == *env_tasks) ? 0/*default*/ : atoi(env_tasks);
      ITYPE *const a = (ITYPE*)libxsmm_malloc((size_t)(lda * ka * sizeof(ITYPE)));
      ITYPE *const b = (ITYPE*)libxsmm_malloc((size_t)(ldb * kb * sizeof(ITYPE)));
      OTYPE *const c = (OTYPE*)libxsmm_malloc((size_t)(ldc * nn * sizeof(OTYPE)));
#if defined(CHECK)
      OTYPE* d = 0;
      if (!LIBXSMM_FEQ(0, check)) {
        d = (OTYPE*)libxsmm_malloc((size_t)(ldc * nn * sizeof(OTYPE)));
        LIBXSMM_MATINIT(OTYPE, 0, d, m, n, ldc, 1.0);
      }
#endif
      LIBXSMM_MATINIT(OTYPE,  0, c,  m,  n, ldc, 1.0);
      LIBXSMM_MATINIT(ITYPE, 42, a, mm, ka, lda, 1.0);
      LIBXSMM_MATINIT(ITYPE, 24, b, kk, kb, ldb, 1.0);
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      /* warm-up OpenMP (populate thread pool) */
#if defined(CHECK)
      if (0 != d) {
        XGEMM_GOLD(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, d, &ldc);
      }
#endif
      XGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(ITYPE),
        &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      fprintf(stdout, "\n\n");

      if (0 == tasks) { /* tiled xGEMM (with library-internal parallelization) */
        int i; double duration;
        unsigned long long start = libxsmm_timer_tick();
        for (i = 0; i < nrepeat; ++i) {
          XGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
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
#         pragma omp parallel
#         pragma omp single nowait
#endif
          XGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
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
          XGEMM_GOLD(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, d, &ldc);
        }
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick());

        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        result = libxsmm_matdiff(LIBXSMM_DATATYPE(OTYPE), m, n, d, c, &ldc, &ldc, &diff);
        if (EXIT_SUCCESS == result) {
          fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
          if (check < diff.l2_rel) {
            fprintf(stderr, "FAILED.\n");
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
  }
  else {
    fprintf(stdout, "Error: invalid argument!\n");
    result = EXIT_FAILURE;
  }
  return result;
}

