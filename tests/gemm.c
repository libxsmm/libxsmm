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
#include <libxsmm_utils.h>
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
#if !defined(GEMM_NO_BYPASS)
# define SMM_NO_BYPASS(FLAGS, ALPHA, BETA) LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA)
#endif
#if (LIBXSMM_EQUAL(REALTYPE, float) || LIBXSMM_EQUAL(REALTYPE, double)) \
  && !defined(MKL_DIRECT_CALL_SEQ) && !defined(MKL_DIRECT_CALL)
LIBXSMM_BLAS_SYMBOL_DECL(REALTYPE, gemm)
#endif


int main(void)
{
  /* test#:                 1  2  3  4  5  6  7  8  9  10 11 12 13    14   15     16  17  18  19  20     21   22   23   24   25   26   27   28   29   30  31  32  33  34    35    36  37 38 */
  /* index:                 0  1  2  3  4  5  6  7  8   9 10 11 12    13   14     15  16  17  18  19     20   21   22   23   24   25   26   27   28   29  30  31  32  33    34    35  36 37 */
  libxsmm_blasint m[]   = { 0, 1, 0, 0, 1, 1, 2, 3, 3,  3, 1, 4, 8,   64,  64,    16, 80, 80, 80, 80,    16, 260, 260, 260, 260, 350, 350, 350, 350, 350,  5, 10, 12, 20,   32,    9, 13, 5 };
  libxsmm_blasint n[]   = { 0, 0, 1, 0, 1, 2, 2, 3, 1, 64, 3, 2, 1,    8, 239, 13824,  1,  3,  5,  7, 65792,   1,   3,   5,   7,  16,   1,  25,   4,   9, 13,  1, 10,  6,   33,    9, 13, 5 };
  libxsmm_blasint k[]   = { 0, 0, 0, 1, 1, 2, 2, 3, 2, 25, 2, 4, 0,   64,  64,    16,  1,  3,  6, 10,    16,   1,   3,   6,  10,  20,   1,  35,   4,  10, 70,  1, 12,  6,  192, 1742, 13, 5 };
  libxsmm_blasint lda[] = { 1, 1, 1, 1, 1, 1, 2, 3, 3,  3, 1, 4, 8,   64,  64,    16, 80, 80, 80, 80,    16, 260, 260, 260, 260, 350, 350, 350, 350, 350,  5, 22, 22, 22,   32,    9, 13, 5 };
  libxsmm_blasint ldb[] = { 1, 1, 1, 1, 1, 2, 2, 3, 2, 25, 2, 4, 8, 9216, 240,    16,  1,  3,  5,  5,    16,   1,   3,   5,   7,  35,  35,  35,  35,  35, 70,  1, 20,  8, 2048, 1742, 13, 5 };
  libxsmm_blasint ldc[] = { 1, 1, 1, 1, 1, 1, 2, 3, 3,  3, 1, 4, 8, 4096, 240,    16, 80, 80, 80, 80,    16, 260, 260, 260, 260, 350, 350, 350, 350, 350,  5, 22, 12, 20, 2048,    9, 13, 5 };
  REALTYPE alpha[]      = { 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1,    1,   1,     1,  1,  1,  1,  1,     1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  1,  1,  1,  1,    1,    1,  1, 2 };
  REALTYPE beta[]       = { 0, 0, 0, 0, 1, 1, 1, 1, 0,  1, 1, 0, 0,    0,   1,     0,  0,  0,  0,  0,     1,   0,   0,   0,   0,   0,   0,   1,   0,   0,  1,  0,  1,  0,    1,    0,  2, 1 };
#if (!defined(__BLAS) || (0 != __BLAS))
  char transa[] = "NNNTT";
#else
  char transa[] = "NN";
#endif
  char transb[] = "NNTNT";
  const int end = sizeof(m) / sizeof(*m);
  const int i0 = 0, i1 = sizeof(transa) - 1;
  libxsmm_blasint max_size_a = 0, max_size_b = 0, max_size_c = 0, block = 1;
#if defined(_DEBUG)
  libxsmm_matdiff_info diff;
#endif
  REALTYPE *a = NULL, *b = NULL, *c = NULL, *d = NULL, *gold = NULL;
  int result = EXIT_SUCCESS, begin = 0, test, i;
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
  d = (REALTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(REALTYPE)));
  gold = (REALTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(REALTYPE)));
  LIBXSMM_ASSERT(NULL != a && NULL != b && NULL != c && NULL != d && NULL != gold);
  LIBXSMM_MATINIT(REALTYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
  LIBXSMM_MATINIT(REALTYPE, 24, b, max_size_b, 1, max_size_b, 1.0);
#if defined(_DEBUG)
  libxsmm_matdiff_clear(&diff);
#endif
#if (defined(__BLAS) && (0 == __BLAS))
  begin = end;
# if defined(_DEBUG)
  fprintf(stderr, "WARNING: skipped tests due to missing BLAS support!\n");
# endif
#endif
  for (test = begin; test < end && EXIT_SUCCESS == result; ++test) {
    for (i = i0; i < i1 && EXIT_SUCCESS == result; ++i) {
      libxsmm_blasint mi = m[test], ni = n[test], ki = k[test];
      const int flags = LIBXSMM_GEMM_FLAGS(transa[i], transb[i]) | \
        ((beta[test] == 0) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0);
      const int no_bypass = SMM_NO_BYPASS(flags, alpha[test], beta[test]);
      const int init = (LIBXSMM_FEQ(0, beta[test]) ? -1 : 0);
      libxsmm_xmmfunction kernel = { NULL };
      libxsmm_gemm_shape gemm_shape;
#if defined(CHECK_FPE) && defined(_MM_GET_EXCEPTION_MASK)
      _MM_SET_EXCEPTION_STATE(0);
#endif
      memset(gold, init, (size_t)(sizeof(REALTYPE) * max_size_c));
      memset(c, init, (size_t)(sizeof(REALTYPE) * max_size_c));
      memset(d, init, (size_t)(sizeof(REALTYPE) * max_size_c));
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
      GEMM(REALTYPE)(transa + i, transb + i, &mi, &ni, &ki,
        alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);
      gemm_shape = libxsmm_create_gemm_shape(
        mi, ni, ki, lda[test], ldb[test], ldc[test],
        LIBXSMM_DATATYPE(REALTYPE), LIBXSMM_DATATYPE(REALTYPE),
        LIBXSMM_DATATYPE(REALTYPE), LIBXSMM_DATATYPE(REALTYPE));
      kernel.gemm = (0 != no_bypass
        ? libxsmm_dispatch_gemm(gemm_shape, flags, LIBXSMM_GEMM_PREFETCH_NONE)
        : NULL);
      if (NULL != kernel.ptr_const) {
        libxsmm_gemm_param gemm_param;
        gemm_param.a.primary = a;
        gemm_param.b.primary = b;
        gemm_param.c.primary = d;
        kernel.gemm(&gemm_param);
      }
#if defined(CHECK_FPE) && defined(_MM_GET_EXCEPTION_MASK)
      fpstate = _MM_GET_EXCEPTION_STATE() & fpcheck;
      result = (0 == fpstate ? EXIT_SUCCESS : EXIT_FAILURE);
      if (EXIT_SUCCESS != result) {
# if defined(_DEBUG)
        fprintf(stderr, "FPE(%i.%i): state=0x%08x -> invalid=%s overflow=%s\n", test + 1, i + 1, fpstate,
          0 != (_MM_MASK_INVALID  & fpstate) ? "true" : "false",
          0 != (_MM_MASK_OVERFLOW & fpstate) ? "true" : "false");
# endif
        break;
      }
#endif
      LIBXSMM_ASSERT(EXIT_SUCCESS == result);
      if (0 == no_bypass || NULL != kernel.ptr_const) {
        libxsmm_matdiff_info diff_test;
        GEMM_GOLD(REALTYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, gold, ldc + test);
        result = libxsmm_matdiff(&diff_test, LIBXSMM_DATATYPE(REALTYPE), mi, ni, gold, c, ldc + test, ldc + test);
        if (EXIT_SUCCESS == result && NULL != kernel.ptr_const) {
          libxsmm_matdiff_info diff_test_kernel;
          result = libxsmm_matdiff(&diff_test_kernel, LIBXSMM_DATATYPE(REALTYPE), mi, ni, gold, d, ldc + test, ldc + test);
          if (EXIT_SUCCESS == result) libxsmm_matdiff_reduce(&diff_test, &diff_test_kernel);
        }
        if (EXIT_SUCCESS == result) {
#if defined(_DEBUG)
          libxsmm_matdiff_reduce(&diff, &diff_test);
#endif
          if (1.0 < (1000.0 * diff_test.normf_rel)) {
#if defined(_DEBUG)
            const char *const target = libxsmm_get_target_arch();
            fprintf(stderr, "\nERROR (%s): test %i.%i failed!\n\t", target, test + 1, i + 1);
            libxsmm_gemm_print(stderr, LIBXSMM_DATATYPE(REALTYPE), transa + i, transb + i, &mi, &ni, &ki,
              alpha + test, NULL/*a*/, lda + test, NULL/*b*/, ldb + test, beta + test, NULL/*c*/, ldc + test);
            fprintf(stderr, "\n");
#endif
            result = EXIT_FAILURE;
          }
          /* avoid drift between Gold and test-results */
          memcpy(c, gold, (size_t)(sizeof(REALTYPE) * max_size_c));
          memcpy(d, gold, (size_t)(sizeof(REALTYPE) * max_size_c));
        }
      }
      else {
#if (0 != LIBXSMM_JIT)
# if defined(_DEBUG)
        fprintf(stderr, "\nERROR: kernel %i.%i not generated!\n\t", test + 1, i + 1);
        libxsmm_gemm_print(stderr, LIBXSMM_DATATYPE(REALTYPE), transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, NULL/*a*/, lda + test, NULL/*b*/, ldb + test, beta + test, NULL/*c*/, ldc + test);
        fprintf(stderr, "\n");
# endif
        result = EXIT_FAILURE;
#endif
      }
    }
  }
#if defined(_DEBUG)
  fprintf(stderr, "Diff (%g vs %g): L2abs=%g Linf=%g Similarity=%g\n",
    diff.v_ref, diff.v_tst, diff.l2_abs, diff.linf_abs, diff.rsq);
#endif
#if defined(CHECK_FPE) && defined(_MM_GET_EXCEPTION_MASK)
  _MM_SET_EXCEPTION_MASK(fpemask); /* restore FPE mask */
  _MM_SET_EXCEPTION_STATE(0); /* clear FPE state */
#endif
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);
  libxsmm_free(d);
  libxsmm_free(gold);
  return result;
}
