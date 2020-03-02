/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(__MKL)
# include <mkl_service.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(ITYPE)
# define ITYPE float
#endif

#if !defined(CHECK) && (LIBXSMM_EQUAL(ITYPE, float) || LIBXSMM_EQUAL(ITYPE, double))
# if !defined(MKL_DIRECT_CALL_SEQ) && !defined(MKL_DIRECT_CALL)
LIBXSMM_BLAS_SYMBOL_DECL(ITYPE, gemm)
# endif
# define CHECK
#endif

#define MYASSERT(x) if (!(x)) { printf("Assertion %s failed...\n", #x); exit(1);}


int main(int argc, char* argv[])
{
  LIBXSMM_BLAS_CONST libxsmm_blasint m = (1 < argc ? atoi(argv[1]) : 1024);
  LIBXSMM_BLAS_CONST libxsmm_blasint k = (3 < argc ? atoi(argv[3]) : m);
  LIBXSMM_BLAS_CONST libxsmm_blasint n = (2 < argc ? atoi(argv[2]) : k);
  const libxsmm_blasint bm = (4 < argc ? atoi(argv[4]) : 32);
  const libxsmm_blasint bk = (6 < argc ? atoi(argv[6]) : bm);
  const libxsmm_blasint bn = (5 < argc ? atoi(argv[5]) : bk);
  const libxsmm_blocked_gemm_order order = (libxsmm_blocked_gemm_order)(7 < argc ? atoi(argv[7]) : 0);
  const int nrepeat = (8 < argc ? atoi(argv[8]) : 100);
  const libxsmm_blasint b_m1 = (9 < argc ? atoi(argv[9]) : 1);
  const libxsmm_blasint b_n1  = (10 < argc ? atoi(argv[10]) : 1);
  const libxsmm_blasint b_k1 = (11 < argc ? atoi(argv[11]) : 1);
  const libxsmm_blasint b_k2 = (12 < argc ? atoi(argv[12]) : 1);
  const int ab = (13 < argc ? atoi(argv[13]) : 0);
  LIBXSMM_BLAS_CONST libxsmm_blasint lda = (14 < argc ? atoi(argv[13]) : m);
  LIBXSMM_BLAS_CONST libxsmm_blasint ldb = (15 < argc ? atoi(argv[14]) : k);
  LIBXSMM_BLAS_CONST libxsmm_blasint ldc = (16 < argc ? atoi(argv[15]) : m);
  LIBXSMM_BLAS_CONST char transa = 'N', transb = 'N'; /* no transposes */
  LIBXSMM_BLAS_CONST ITYPE alpha = 1, beta = 1;
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const double gflops = 2.0 * m * n * k * 1E-9;
  int result = EXIT_SUCCESS;
#if defined(CHECK) && (!defined(__BLAS) || (0 != __BLAS))
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(NULL == env_check ? 0 : atof(env_check));
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
    ITYPE* agold = (ITYPE*)libxsmm_malloc((size_t)lda * (size_t)k * sizeof(ITYPE));
    ITYPE* bgold = (ITYPE*)libxsmm_malloc((size_t)ldb * (size_t)n * sizeof(ITYPE));
    ITYPE* cgold = (ITYPE*)libxsmm_malloc((size_t)ldc * (size_t)n * sizeof(ITYPE));
    ITYPE* a = (ITYPE*)libxsmm_malloc((size_t)m * (size_t)k * sizeof(ITYPE));
    ITYPE* b = (ITYPE*)libxsmm_malloc((size_t)k * (size_t)n * sizeof(ITYPE));
    ITYPE* c = (ITYPE*)libxsmm_malloc((size_t)m * (size_t)n * sizeof(ITYPE));
    libxsmm_blocked_gemm_handle* handle = 0;
    unsigned long long start;
    double duration;
#if defined(_OPENMP)
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    handle = libxsmm_blocked_gemm_handle_create(nthreads,
      LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, NULL/*auto-prefetch*/, &order);

    if (0 != handle) {
      LIBXSMM_MATINIT_OMP(ITYPE, 42, agold, m, k, lda, 1.0);
      LIBXSMM_MATINIT_OMP(ITYPE, 24, bgold, k, n, ldb, 1.0);
      LIBXSMM_MATINIT_OMP(ITYPE,  0, cgold, m, n, ldc, 1.0);
      libxsmm_blocked_gemm_copyin_a(handle, agold, &lda, a);
      libxsmm_blocked_gemm_copyin_b(handle, bgold, &ldb, b);
      libxsmm_blocked_gemm_copyin_c(handle, cgold, &ldc, c);
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      /* warm-up OpenMP (populate thread pool) */
      libxsmm_blocked_gemm_omp(handle, a, b, c, 1);
#if defined(CHECK) && (!defined(__BLAS) || (0 != __BLAS))
      if (!LIBXSMM_FEQ(0, check)) {
        LIBXSMM_GEMM_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, agold, &lda, bgold, &ldb, &beta, cgold, &ldc);
      }
#endif
      if (!ab) {
      libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(ITYPE),
        &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      fprintf(stdout, "\n\n");
      }
      start = libxsmm_timer_tick();
      libxsmm_blocked_gemm_omp(handle, a, b, c, nrepeat);
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        if (ab) {
          fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s | %lli,%lli,%lli,%lli,%lli,%lli,%i,%lli,%lli,%lli,%lli\n",
            gflops * nrepeat / duration, (long long)m, (long long)n, (long long)k, (long long)bm, (long long)bn, (long long)bk,
            (int)order, (long long)b_m1, (long long)b_n1, (long long)b_k1, (long long)b_k2);
        } else {
          fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
      }
#if defined(CHECK) && (!defined(__BLAS) || (0 != __BLAS))
      if (!LIBXSMM_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        ITYPE* ctest = 0;
        int i;
        start = libxsmm_timer_tick();
        for (i = 0; i < nrepeat; ++i) {
          LIBXSMM_GEMM_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, agold, &lda, bgold, &ldb, &beta, cgold, &ldc);
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
        ctest = (ITYPE*)libxsmm_malloc((size_t)(sizeof(ITYPE) * ldc * n));
        if (0 != ctest) {
          libxsmm_matdiff_info diff;
          libxsmm_blocked_gemm_copyout_c(handle, c, &ldc, ctest);
          result = libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(ITYPE), m, n, cgold, ctest, &ldc, &ldc);
          if (EXIT_SUCCESS == result) {
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
      libxsmm_blocked_gemm_handle_destroy(handle);
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
  if (!ab) {
    fprintf(stdout, "Finished\n");
  }
  return result;
}

