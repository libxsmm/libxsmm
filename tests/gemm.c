/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <utils/libxsmm_utils.h>
#include <libxsmm.h>

#if !defined(REALTYPE)
# define REALTYPE double
#endif
#if !defined(CHECK_FPE)
# define CHECK_FPE
#endif
#if !defined(GEMM_GOLD)
# define GEMM_GOLD LIBXSMM_GEMM_SYMBOL
#endif
#if !defined(GEMM)
# define GEMM(TYPE) LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, gemm))
#endif
#if !defined(SMM)
# define SMM(TYPE) LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, gemm))
#endif
#if !defined(GEMM_NO_BYPASS)
# define SMM_NO_BYPASS(FLAGS, ALPHA, BETA) LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA)
#endif
#if (LIBXSMM_EQUAL(REALTYPE, float) || LIBXSMM_EQUAL(REALTYPE, double)) \
  && !defined(MKL_DIRECT_CALL_SEQ) && !defined(MKL_DIRECT_CALL)
LIBXSMM_BLAS_SYMBOL_DECL(REALTYPE, gemm)
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
  REALTYPE alpha[]      = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,    1,   1,     1,  1,  1,  1,  1,     1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  1,  1,  1,  1,    1,    1,  1, 1 };
  REALTYPE beta[]       = { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,    0,   1,     0,  0,  0,  0,  0,     1,   0,   0,   0,   0,   0,   0,   1,   0,   0,  1,  0,  1,  0,    1,    0,  1, 1 };
#if defined(LIBXSMM_PLATFORM_X86) && (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
  char transa[] = "NNNTT";
#else
  char transa[] = "NN";
#endif
#if defined(LIBXSMM_PLATFORM_X86)
  char transb[] = "NNTNT";
  const int begin = 0;
#else
  char transb[] = "NN";
  const int begin = 4;
#endif
  const int end = sizeof(m) / sizeof(*m);
  const int i0 = 0, i1 = sizeof(transa) - 1;
  libxsmm_blasint max_size_a = 0, max_size_b = 0, max_size_c = 0, block = 1;
#if defined(_DEBUG)
  libxsmm_matdiff_info diff;
#endif
  REALTYPE *a = NULL, *b = NULL;
  REALTYPE *c = NULL;
#if defined(GEMM)
  REALTYPE *d = NULL;
#endif
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
  REALTYPE *gold = NULL;
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
  a = (REALTYPE*)libxsmm_malloc((size_t)(max_size_a * sizeof(REALTYPE)));
  b = (REALTYPE*)libxsmm_malloc((size_t)(max_size_b * sizeof(REALTYPE)));
  c = (REALTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(REALTYPE)));
#if defined(GEMM)
  d = (REALTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(REALTYPE)));
  LIBXSMM_ASSERT(NULL != d);
#endif
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
  gold = (REALTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(REALTYPE)));
  LIBXSMM_ASSERT(NULL != gold);
#endif
  LIBXSMM_ASSERT(NULL != a && NULL != b && NULL != c);
  LIBXSMM_MATINIT(REALTYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
  LIBXSMM_MATINIT(REALTYPE, 24, b, max_size_b, 1, max_size_b, 1.0);
#if defined(_DEBUG)
  libxsmm_matdiff_clear(&diff);
#endif
  for (test = begin; test < end && EXIT_SUCCESS == result; ++test) {
    for (i = i0; i < i1 && EXIT_SUCCESS == result; ++i) {
      libxsmm_blasint mi = m[test], ni = n[test], ki = k[test];
      const int flags = LIBXSMM_GEMM_FLAGS(transa[i], transb[i]) | ((beta[i] == 0) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0);
      const int smm = SMM_NO_BYPASS(flags, alpha[test], beta[test]);
#if (0 == LIBXSMM_JIT)
      LIBXSMM_UNUSED(flags);
#endif
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
        memset(gold, -1, (size_t)(sizeof(REALTYPE) * max_size_c));
#endif
        memset(c, -1, (size_t)(sizeof(REALTYPE) * max_size_c));
#if defined(GEMM)
        memset(d, -1, (size_t)(sizeof(REALTYPE) * max_size_c));
#endif
      }
      else {
#if (!defined(__BLAS) || (0 != __BLAS)) && defined(GEMM_GOLD)
        memset(gold, 0, (size_t)(sizeof(REALTYPE) * max_size_c));
#endif
        memset(c, 0, (size_t)(sizeof(REALTYPE) * max_size_c));
#if defined(GEMM)
        memset(d, 0, (size_t)(sizeof(REALTYPE) * max_size_c));
#endif
      }
      if (0 != smm) {
        SMM(REALTYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);
      }
#if defined(GEMM)
      else {
        GEMM(REALTYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);
      }
      GEMM(REALTYPE)(transa + i, transb + i, &mi, &ni, &ki,
        alpha + test, a, lda + test, b, ldb + test, beta + test, d, ldc + test);
#endif
#if (0 != LIBXSMM_JIT)
      if (0 != smm) { /* dispatch kernel and check that it is available */
        libxsmm_xmmfunction kernel = { NULL };
        const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
          mi, ni, ki, lda[test], ldb[test], ldc[test],
          LIBXSMM_DATATYPE(REALTYPE), LIBXSMM_DATATYPE(REALTYPE),
          LIBXSMM_DATATYPE(REALTYPE), LIBXSMM_DATATYPE(REALTYPE));
        kernel.gemm = libxsmm_dispatch_gemm_v2(gemm_shape, flags, LIBXSMM_PREFETCH_NONE);
        if (NULL == kernel.ptr_const) {
# if defined(_DEBUG)
          fprintf(stderr, "\nERROR: kernel %i.%i not generated!\n\t", test + 1, i + 1);
          libxsmm_gemm_print(stderr, LIBXSMM_DATATYPE(REALTYPE), transa + i, transb + i, &mi, &ni, &ki,
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
        GEMM_GOLD(REALTYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, gold, ldc + test);

        result = libxsmm_matdiff(&diff_test, LIBXSMM_DATATYPE(REALTYPE), mi, ni, gold, c, ldc + test, ldc + test);
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
            libxsmm_gemm_print(stderr, LIBXSMM_DATATYPE(REALTYPE), transa + i, transb + i, &mi, &ni, &ki,
              alpha + test, NULL/*a*/, lda + test, NULL/*b*/, ldb + test, beta + test, NULL/*c*/, ldc + test);
            fprintf(stderr, "\n");
#   endif
            result = EXIT_FAILURE;
          }
#   if defined(GEMM)
          else {
            result = libxsmm_matdiff(&diff_test, LIBXSMM_DATATYPE(REALTYPE), mi, ni, gold, d, ldc + test, ldc + test);
            if (EXIT_SUCCESS == result) {
#     if defined(_DEBUG)
              libxsmm_matdiff_reduce(&diff, &diff_test);
#     endif
              if (1.0 < (1000.0 * diff_test.normf_rel)) {
#     if defined(_DEBUG)
                fprintf(stderr, "\nERROR: test %i.%i failed!\n\t", test + 1, i + 1);
                libxsmm_gemm_print(stderr, LIBXSMM_DATATYPE(REALTYPE), transa + i, transb + i, &mi, &ni, &ki,
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
      memcpy(c, gold, (size_t)(sizeof(REALTYPE) * max_size_c));
#   if defined(GEMM)
      memcpy(d, gold, (size_t)(sizeof(REALTYPE) * max_size_c));
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
