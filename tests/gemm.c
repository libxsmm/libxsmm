/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>

#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif
#if !defined(CHECK_FPE)
# define CHECK_FPE
#endif
#if !defined(GEMM_GOLD)
# define GEMM_GOLD LIBXSMM_GEMM_SYMBOL
#endif
#if !defined(GEMM)
# define GEMM LIBXSMM_XGEMM_SYMBOL
#endif
#if !defined(GEMM2)
# define GEMM2 LIBXSMM_YGEMM_SYMBOL
#endif
#if !defined(SMM)
# define SMM LIBXSMM_XGEMM_SYMBOL
#endif
#if !defined(GEMM_NO_BYPASS)
# define SMM_NO_BYPASS(FLAGS, ALPHA, BETA) LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA)
#endif


#if (LIBXSMM_EQUAL(ITYPE, float) || LIBXSMM_EQUAL(ITYPE, double)) \
  && !defined(MKL_DIRECT_CALL_SEQ) && !defined(MKL_DIRECT_CALL)
LIBXSMM_BLAS_SYMBOL_DECL(ITYPE, gemm)
#endif


int main(void)
{
  /* test#:                 1  2  3  4  5  6  7  8  9 10 11 12    13   14     15  16  17  18  19     20   21   22   23   24   25   26   27   28   29  30  31  32  33    34    35  36 37 */
  /* index:                 0  1  2  3  4  5  6  7  8  9 10 11    12   13     14  15  16  17  18     19   20   21   22   23   24   25   26   27   28  29  30  31  32    33    34  35 36 */
  libxsmm_blasint m[]   = { 0, 1, 0, 0, 1, 1, 2, 3, 3, 1, 4, 8,   64,  64,    16, 80, 80, 80, 80,    16, 260, 260, 260, 260, 350, 350, 350, 350, 350,  5, 10, 12, 20,   32,    9, 13, 5 };
  libxsmm_blasint n[]   = { 0, 0, 1, 0, 1, 2, 2, 3, 1, 3, 1, 1,    8, 239, 13824,  1,  3,  5,  7, 65792,   1,   3,   5,   7,  16,   1,  25,   4,   9, 13,  1, 10,  6,   33,    9, 13, 5 };
  libxsmm_blasint k[]   = { 0, 0, 0, 1, 1, 2, 2, 3, 2, 2, 4, 0,   64,  64,    16,  1,  3,  6, 10,    16,   1,   3,   6,  10,  20,   1,  35,   4,  10, 70,  1, 12,  6,  192, 1742, 13, 5 };
  libxsmm_blasint lda[] = { 1, 1, 1, 1, 1, 1, 2, 3, 3, 1, 4, 8,   64,  64,    16, 80, 80, 80, 80,    16, 260, 260, 260, 260, 350, 350, 350, 350, 350,  5, 22, 22, 22,   32,    9, 13, 5 };
  libxsmm_blasint ldb[] = { 1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 4, 8, 9216, 240,    16,  1,  3,  5,  5,    16,   1,   3,   5,   7,  35,  35,  35,  35,  35, 70,  1, 20,  8, 2048, 1742, 13, 5 };
  libxsmm_blasint ldc[] = { 1, 1, 1, 1, 1, 1, 2, 3, 3, 1, 4, 8, 4096, 240,    16, 80, 80, 80, 80,    16, 260, 260, 260, 260, 350, 350, 350, 350, 350,  5, 22, 12, 20, 2048,    9, 13, 5 };
  OTYPE alpha[]         = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,    1,   1,     1,  1,  1,  1,  1,     1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  1,  1,  1,  1,    1,    1,  1, 1 };
  OTYPE beta[]          = { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,    0,   1,     0,  0,  0,  0,  0,     1,   0,   0,   0,   0,   0,   0,   1,   0,   0,  1,  0,  1,  0,    1,    0,  1, 1 };
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
  char transa[] = "NNNTT";
#else
  char transa[] = "NN";
#endif
  char transb[] = "NNTNT";
  const int begin = 0, end = sizeof(m) / sizeof(*m), i0 = 0, i1 = sizeof(transa) - 1;
  libxsmm_blasint max_size_a = 0, max_size_b = 0, max_size_c = 0, block = 1;
#if defined(_DEBUG)
  libxsmm_matdiff_info diff;
#endif
  ITYPE *a = NULL, *b = NULL;
  OTYPE *c = NULL;
#if defined(GEMM)
  OTYPE *d = NULL;
#endif
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
  OTYPE *gold = NULL;
#endif
  int result = EXIT_SUCCESS, test, i;
#if defined(CHECK_FPE) && defined(_MM_GET_EXCEPTION_MASK)
  const unsigned int fpemask = _MM_GET_EXCEPTION_MASK(); /* backup FPE mask */
  const unsigned int fpcheck = _MM_MASK_INVALID | _MM_MASK_OVERFLOW;
  unsigned int fpstate = 0;
  _MM_SET_EXCEPTION_MASK(fpemask & ~fpcheck);
#endif
  LIBXSMM_BLAS_INIT
  for (test = begin; test < end; ++test) {
    m[test] = LIBXSMM_UP(m[test], block);
    n[test] = LIBXSMM_UP(n[test], block);
    k[test] = LIBXSMM_UP(k[test], block);
    lda[test] = LIBXSMM_MAX(lda[test], m[test]);
    ldb[test] = LIBXSMM_MAX(ldb[test], k[test]);
    ldc[test] = LIBXSMM_MAX(ldc[test], m[test]);
  }
  for (test = begin; test < end; ++test) {
    const libxsmm_blasint size_a = lda[test] * k[test], size_b = ldb[test] * n[test], size_c = ldc[test] * n[test];
    LIBXSMM_ASSERT(m[test] <= lda[test] && k[test] <= ldb[test] && m[test] <= ldc[test]);
    max_size_a = LIBXSMM_MAX(max_size_a, size_a);
    max_size_b = LIBXSMM_MAX(max_size_b, size_b);
    max_size_c = LIBXSMM_MAX(max_size_c, size_c);
  }
  a = (ITYPE*)libxsmm_malloc((size_t)(max_size_a * sizeof(ITYPE)));
  b = (ITYPE*)libxsmm_malloc((size_t)(max_size_b * sizeof(ITYPE)));
  c = (OTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(OTYPE)));
#if defined(GEMM)
  d = (OTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(OTYPE)));
  LIBXSMM_ASSERT(NULL != d);
#endif
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
  gold = (OTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(OTYPE)));
  LIBXSMM_ASSERT(NULL != gold);
#endif
  LIBXSMM_ASSERT(NULL != a && NULL != b && NULL != c);
  LIBXSMM_MATINIT(ITYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
  LIBXSMM_MATINIT(ITYPE, 24, b, max_size_b, 1, max_size_b, 1.0);
#if defined(_DEBUG)
  libxsmm_matdiff_clear(&diff);
#endif
  for (test = begin; test < end && EXIT_SUCCESS == result; ++test) {
    for (i = i0; i < i1 && EXIT_SUCCESS == result; ++i) {
      libxsmm_blasint mi = m[test], ni = n[test], ki = k[test];
      const int flags = LIBXSMM_GEMM_FLAGS(transa[i], transb[i]);
      const int smm = SMM_NO_BYPASS(flags, alpha[test], beta[test]);
#if defined(CHECK_FPE) && defined(_MM_GET_EXCEPTION_MASK)
      _MM_SET_EXCEPTION_STATE(0);
#endif
      if ('N' != transa[i] && 'N' == transb[i]) { /* TN */
        mi = ki = LIBXSMM_MIN(mi, ki);
      }
      else if ('N' == transa[i] && 'N' != transb[i]) { /* NT */
        ki = ni = LIBXSMM_MIN(ki, ni);
      }
      else if ('N' != transa[i] && 'N' != transb[i]) { /* TT */
        const libxsmm_blasint ti = LIBXSMM_MIN(mi, ni);
        mi = ni = ki = LIBXSMM_MIN(ti, ki);
      }
      if (LIBXSMM_FEQ(0, beta[test])) {
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
        memset(gold, -1, (size_t)(sizeof(OTYPE) * max_size_c));
#endif
        memset(c, -1, (size_t)(sizeof(OTYPE) * max_size_c));
#if defined(GEMM)
        memset(d, -1, (size_t)(sizeof(OTYPE) * max_size_c));
#endif
      }
      else {
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
        memset(gold, 0, (size_t)(sizeof(OTYPE) * max_size_c));
#endif
        memset(c, 0, (size_t)(sizeof(OTYPE) * max_size_c));
#if defined(GEMM)
        memset(d, 0, (size_t)(sizeof(OTYPE) * max_size_c));
#endif
      }
      if (0 != smm) {
        SMM(ITYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);
      }
#if defined(GEMM)
      else {
        GEMM(ITYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);
      }
# if defined(GEMM2)
      GEMM2(ITYPE)(transa + i, transb + i, &mi, &ni, &ki,
        alpha + test, a, lda + test, b, ldb + test, beta + test, d, ldc + test);
# else
      GEMM(ITYPE)(transa + i, transb + i, &mi, &ni, &ki,
        alpha + test, a, lda + test, b, ldb + test, beta + test, d, ldc + test);
# endif
#endif
#if (0 != LIBXSMM_JIT)
      if (0 != smm) { /* dispatch kernel and check that it is available */
        const LIBXSMM_MMFUNCTION_TYPE(ITYPE) kernel = LIBXSMM_MMDISPATCH_SYMBOL(ITYPE)(mi, ni, ki,
          lda + test, ldb + test, ldc + test, alpha + test, beta + test, &flags, NULL/*prefetch*/);
        if (NULL == kernel) {
# if defined(_DEBUG)
          fprintf(stderr, "\nERROR: kernel %i.%i not generated!\n\t", test + 1, i + 1);
          libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION(ITYPE), transa + i, transb + i, &mi, &ni, &ki,
            alpha + test, NULL/*a*/, lda + test, NULL/*b*/, ldb + test, beta + test, NULL/*c*/, ldc + test);
          fprintf(stderr, "\n");
# endif
          result = EXIT_FAILURE;
          break;
        }
      }
#endif
#if defined(CHECK_FPE) && defined(_MM_GET_EXCEPTION_MASK)
      fpstate = _MM_GET_EXCEPTION_STATE() & fpcheck;
      result = (0 == fpstate ? EXIT_SUCCESS : EXIT_FAILURE);
      if (EXIT_SUCCESS != result) {
# if defined(_DEBUG)
        fprintf(stderr, "FPE(%i.%i): state=0x%08x -> invalid=%s overflow=%s\n", test + 1, i + 1, fpstate,
          0 != (_MM_MASK_INVALID  & fpstate) ? "true" : "false",
          0 != (_MM_MASK_OVERFLOW & fpstate) ? "true" : "false");
# endif
      }
# if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
      else
# endif
#endif
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
# if !defined(GEMM)
      if (0 != smm)
# endif
      {
# if defined(GEMM_GOLD)
        libxsmm_matdiff_info diff_test;
        GEMM_GOLD(ITYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, gold, ldc + test);

        result = libxsmm_matdiff(&diff_test, LIBXSMM_DATATYPE(OTYPE), mi, ni, gold, c, ldc + test, ldc + test);
        if (EXIT_SUCCESS == result) {
#   if defined(_DEBUG)
          libxsmm_matdiff_reduce(&diff, &diff_test);
#   endif
          if (1.0 < (1000.0 * diff_test.normf_rel)) {
#   if defined(_DEBUG)
            if (0 != smm) {
              fprintf(stderr, "\nERROR: SMM test %i.%i failed!\n\t", test + 1, i + 1);
            }
            else {
              fprintf(stderr, "\nERROR: test %i.%i failed!\n\t", test + 1, i + 1);
            }
            libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION(ITYPE), transa + i, transb + i, &mi, &ni, &ki,
              alpha + test, NULL/*a*/, lda + test, NULL/*b*/, ldb + test, beta + test, NULL/*c*/, ldc + test);
            fprintf(stderr, "\n");
#   endif
            result = EXIT_FAILURE;
          }
#   if defined(GEMM)
          else {
            result = libxsmm_matdiff(&diff_test, LIBXSMM_DATATYPE(OTYPE), mi, ni, gold, d, ldc + test, ldc + test);
            if (EXIT_SUCCESS == result) {
#     if defined(_DEBUG)
              libxsmm_matdiff_reduce(&diff, &diff_test);
#     endif
              if (1.0 < (1000.0 * diff_test.normf_rel)) {
#     if defined(_DEBUG)
                fprintf(stderr, "\nERROR: test %i.%i failed!\n\t", test + 1, i + 1);
                libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION(ITYPE), transa + i, transb + i, &mi, &ni, &ki,
                  alpha + test, NULL/*a*/, lda + test, NULL/*b*/, ldb + test, beta + test, NULL/*c*/, ldc + test);
                fprintf(stderr, "\n");
#     endif
                result = EXIT_FAILURE;
              }
            }
          }
#   endif
        }
# endif
      }
# if defined(GEMM_GOLD)
      /* avoid drift between Gold and test-results */
      memcpy(c, gold, (size_t)(sizeof(OTYPE) * max_size_c));
#   if defined(GEMM)
      memcpy(d, gold, (size_t)(sizeof(OTYPE) * max_size_c));
#   endif
# endif
#elif defined(_DEBUG)
      fprintf(stderr, "Warning: skipped the test due to missing BLAS support!\n");
#endif
    }
  }

#if defined(_DEBUG)
  fprintf(stderr, "Diff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
#endif
#if defined(CHECK_FPE) && defined(_MM_GET_EXCEPTION_MASK)
  _MM_SET_EXCEPTION_MASK(fpemask); /* restore FPE mask */
  _MM_SET_EXCEPTION_STATE(0); /* clear FPE state */
#endif
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);
#if defined(GEMM)
  libxsmm_free(d);
#endif
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
  libxsmm_free(gold);
#endif
  return result;
}

