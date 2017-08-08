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
# define REAL_TYPE float
#endif
#if !defined(CHECK) && \
  (!defined(__BLAS) || (0 != __BLAS)) && /* BLAS available */ \
  (LIBXSMM_EQUAL(REAL_TYPE, float) || LIBXSMM_EQUAL(REAL_TYPE, double))
# define CHECK
#endif

#define MYASSERT(x) if(!(x)) { printf("Assertion %s failed...\n", #x); exit(1);}


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
  const libxsmm_blasint m = (1 < argc ? atoi(argv[1]) : 1024);
  const libxsmm_blasint k = (3 < argc ? atoi(argv[3]) : m);
  const libxsmm_blasint n = (2 < argc ? atoi(argv[2]) : k);
  const libxsmm_blasint bm = (4 < argc ? atoi(argv[4]) : 32);
  const libxsmm_blasint bk = (6 < argc ? atoi(argv[6]) : bm);
  const libxsmm_blasint bn = (5 < argc ? atoi(argv[5]) : bk);
  const libxsmm_bgemm_order order = (libxsmm_bgemm_order)(7 < argc ? atoi(argv[7]) : 0);
  const int nrepeat = (8 < argc ? atoi(argv[8]) : 100);
  const libxsmm_blasint b_m1 = (9 < argc ? atoi(argv[9]) : 1);
  const libxsmm_blasint b_n1  = (10 < argc ? atoi(argv[10]) : 1);
  const libxsmm_blasint b_k1 = (11 < argc ? atoi(argv[11]) : 1);
  const libxsmm_blasint b_k2 = (12 < argc ? atoi(argv[12]) : 1);
  const int ab = (13 < argc ? atoi(argv[13]) : 0);
  const libxsmm_blasint lda = (14 < argc ? atoi(argv[13]) : m);
  const libxsmm_blasint ldb = (15 < argc ? atoi(argv[14]) : k);
  const libxsmm_blasint ldc = (16 < argc ? atoi(argv[15]) : m);
  const double gflops = 2.0 * m * n * k * 1E-9;
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;
  int result = EXIT_SUCCESS;
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) { /* check command line */
    printf("\nUsage: ./bgemm [M] [N] [K] [bm] [bn] [bk] [order] [reps] [b_m1] [b_n1] [b_k1] [b_k2] [verbose]\n\n");
    return result;
  }

  MYASSERT(m % b_m1 == 0);
  MYASSERT(n % b_n1 == 0);
  MYASSERT(k % b_k1 == 0);
  MYASSERT(m/b_m1 % bm == 0);
  MYASSERT(n/b_n1 % bn == 0);
  MYASSERT(k/b_k1/b_k2 % bk == 0);

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    REAL_TYPE* agold = (REAL_TYPE*)libxsmm_malloc(lda * k * sizeof(REAL_TYPE));
    REAL_TYPE* bgold = (REAL_TYPE*)libxsmm_malloc(ldb * n * sizeof(REAL_TYPE));
    REAL_TYPE* cgold = (REAL_TYPE*)libxsmm_malloc(ldc * n * sizeof(REAL_TYPE));
    REAL_TYPE* a = (REAL_TYPE*)libxsmm_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* b = (REAL_TYPE*)libxsmm_malloc(k * n * sizeof(REAL_TYPE));
    REAL_TYPE* c = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    libxsmm_bgemm_handle* handle = 0;
    unsigned long long start;
    double duration;
    const libxsmm_gemm_prefetch_type strategy = LIBXSMM_PREFETCH_AUTO;
    handle = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);

    if (0 != handle) {
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
#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) {
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, agold, &lda, bgold, &ldb, &beta, cgold, &ldc);
      }
#endif
      if (!ab) {
      libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(REAL_TYPE),
        &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      fprintf(stdout, "\n\n");
      }
      start = libxsmm_timer_tick();
      libxsmm_bgemm_omp(handle, a, b, c, nrepeat);
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        if (ab) {
          fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s | %d,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i\n",
            gflops * nrepeat / duration, m, n, k, bm, bn, bk, order, b_m1, b_n1, b_k1, b_k2);
        } else {
          fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
      }
#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        REAL_TYPE* ctest = 0;
        int i;
        start = libxsmm_timer_tick();
        for (i = 0; i < nrepeat; ++i) {
          LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, agold, &lda, bgold, &ldb, &beta, cgold, &ldc);
        }
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxsmm_free(agold); agold = 0;
        libxsmm_free(bgold); bgold = 0;
        libxsmm_free(a); a = 0;
        libxsmm_free(b); b = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        ctest = (REAL_TYPE*)libxsmm_malloc(ldc * n * sizeof(REAL_TYPE));
        if (0 != ctest) {
          libxsmm_matdiff_info diff;
          libxsmm_bgemm_copyout_c(handle, c, &ldc, ctest);
          if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(REAL_TYPE), m, n, cgold, ctest, &ldc, &ldc, &diff)) {
            fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
            if (check < 100.0 * diff.normf_rel) {
              fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
              result = EXIT_FAILURE;
            }
          }
          libxsmm_free(ctest);
        }
      }
#endif
      libxsmm_bgemm_handle_destroy(handle);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXSMM_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxsmm_free(agold);
    libxsmm_free(bgold);
    libxsmm_free(cgold);
    libxsmm_free(a);
    libxsmm_free(b);
    libxsmm_free(c);
  }
  if(!ab) {
    fprintf(stdout, "Finished\n");
  }
  return result;
}

