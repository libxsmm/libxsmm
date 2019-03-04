/******************************************************************************
** Copyright (c) 2015-2019, Intel Corporation                                **
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
#include <libxsmm_intrinsics_x86.h>

#include <stdlib.h>
#include <string.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif

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
# define GEMM LIBXSMM_YGEMM_SYMBOL
#endif
#if !defined(SMM)
# define SMM LIBXSMM_XGEMM_SYMBOL
#endif
#if !defined(GEMM_NO_BYPASS)
# define SMM_NO_BYPASS(FLAGS, ALPHA, BETA) (LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA) && \
  0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_B)))
#endif


LIBXSMM_GEMM_SYMBOL_DECL(LIBXSMM_GEMM_CONST, ITYPE)


int main(void)
{
  /* reported test #:       1  2  3  4  5  6  7  8  9 10 11    12   13     14     15   16   17   18   19   20  21  22  23  24    25    26 */
  libxsmm_blasint m[]   = { 0, 1, 0, 0, 1, 1, 2, 3, 3, 1, 8,   64,  64,    16,    16, 350, 350, 350, 350, 350,  5, 10, 12, 20,   32,    9 };
  libxsmm_blasint n[]   = { 0, 0, 1, 0, 1, 2, 2, 3, 1, 3, 1,    8, 239, 13824, 65792,  16,   1,  25,   4,   9, 13,  1, 10,  6,   33,    9 };
  libxsmm_blasint k[]   = { 0, 0, 0, 1, 1, 2, 2, 3, 2, 2, 0,   64,  64,    16,    16,  20,   1,  35,   4,  10, 70,  1, 12,  6,  192, 1742 };
  libxsmm_blasint lda[] = { 1, 1, 1, 1, 1, 1, 2, 3, 3, 1, 8,   64,  64,    16,    16, 350, 350, 350, 350, 350,  5, 22, 22, 22,   32,    9 };
  libxsmm_blasint ldb[] = { 1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 8, 9216, 240,    16,    16,  35,  35,  35,  35,  35, 70,  1, 20,  8, 2048, 1742 };
  libxsmm_blasint ldc[] = { 1, 1, 1, 1, 1, 1, 2, 3, 3, 1, 8, 4096, 240,    16,    16, 350, 350, 350, 350, 350,  5, 22, 12, 20, 2048,    9 };
  OTYPE alpha[]         = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,    1,   1,     1,     1,   1,   1,   1,   1,   1,  1,  1,  1,  1,    1,    1 };
  OTYPE beta[]          = { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,    0,   1,     0,     1,   0,   0,   1,   0,   0,  1,  0,  1,  0,    1,    0 };
#if !defined(__BLAS) || (0 != __BLAS)
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
  OTYPE *c = NULL, *d = NULL;
#if !defined(__BLAS) || (0 != __BLAS)
  OTYPE *gold = NULL;
#endif
  int result = EXIT_SUCCESS, test, i;
#if defined(CHECK_FPE) && defined(_MM_GET_EXCEPTION_MASK)
  const unsigned int fpemask = _MM_GET_EXCEPTION_MASK(); /* backup FPE mask */
  const unsigned int fpcheck = _MM_MASK_INVALID | _MM_MASK_OVERFLOW;
  unsigned int fpstate = 0;
  _MM_SET_EXCEPTION_MASK(fpemask & ~fpcheck);
#endif
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
  d = (OTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(OTYPE)));
#if !defined(__BLAS) || (0 != __BLAS)
  gold = (OTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(OTYPE)));
  LIBXSMM_ASSERT(NULL != gold);
  LIBXSMM_MATINIT_OMP(OTYPE, 0, gold, max_size_c, 1, max_size_c, 1.0);
#endif
  LIBXSMM_ASSERT(NULL != a && NULL != b && NULL != c && NULL != d);
  LIBXSMM_MATINIT_OMP(ITYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
  LIBXSMM_MATINIT_OMP(ITYPE, 24, b, max_size_b, 1, max_size_b, 1.0);
  LIBXSMM_MATINIT_OMP(OTYPE,  0, c, max_size_c, 1, max_size_c, 1.0);
  LIBXSMM_MATINIT_OMP(OTYPE,  0, d, max_size_c, 1, max_size_c, 1.0);
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
      if (0 != smm) {
        SMM(ITYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);
      }
      else {
        GEMM(ITYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);
      }
      GEMM(ITYPE)(transa + i, transb + i, &mi, &ni, &ki,
        alpha + test, a, lda + test, b, ldb + test, beta + test, d, ldc + test);
#if defined(CHECK_FPE) && defined(_MM_GET_EXCEPTION_MASK)
      fpstate = _MM_GET_EXCEPTION_STATE() & fpcheck;
      result = (0 == fpstate ? EXIT_SUCCESS : EXIT_FAILURE);
      if (EXIT_SUCCESS != result) {
# if defined(_DEBUG)
        fprintf(stderr, "FPE(#%i): state=0x%08x -> invalid=%s overflow=%s\n", test + 1, fpstate,
          0 != (_MM_MASK_INVALID  & fpstate) ? "true" : "false",
          0 != (_MM_MASK_OVERFLOW & fpstate) ? "true" : "false");
# endif
      }
# if !defined(__BLAS) || (0 != __BLAS)
      else
# endif
#endif
#if !defined(__BLAS) || (0 != __BLAS)
      {
        libxsmm_matdiff_info diff_test;
        GEMM_GOLD(ITYPE)(transa + i, transb + i, &mi, &ni, &ki,
          alpha + test, a, lda + test, b, ldb + test, beta + test, gold, ldc + test);

        result = libxsmm_matdiff(&diff_test, LIBXSMM_DATATYPE(OTYPE), m[test], n[test], gold, c, ldc + test, ldc + test);
        if (EXIT_SUCCESS == result) {
# if defined(_DEBUG)
          libxsmm_matdiff_reduce(&diff, &diff_test);
# endif
          if (1.0 < (1000.0 * diff_test.normf_rel)) {
# if defined(_DEBUG)
            if (0 != smm) {
              fprintf(stderr, "\nERROR: SMM test %i.%i failed!\n\t", test + 1, i + 1);
            }
            else {
              fprintf(stderr, "\nERROR: test %i.%i failed!\n\t", test + 1, i + 1);
            }
            libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION(ITYPE), transa + i, transb + i, &mi, &ni, &ki,
              alpha + test, NULL/*a*/, lda + test, NULL/*b*/, ldb + test, beta + test, NULL/*c*/, ldc + test);
            fprintf(stderr, "\n");
# endif
            result = EXIT_FAILURE;
          }
          else {
            result = libxsmm_matdiff(&diff_test, LIBXSMM_DATATYPE(OTYPE), m[test], n[test], gold, d, ldc + test, ldc + test);
            if (EXIT_SUCCESS == result) {
# if defined(_DEBUG)
              libxsmm_matdiff_reduce(&diff, &diff_test);
# endif
              if (1.0 < (1000.0 * diff_test.normf_rel)) {
# if defined(_DEBUG)
                fprintf(stderr, "\nERROR: test %i.%i failed!\n\t", test + 1, i + 1);
                libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION(ITYPE), transa + i, transb + i, &mi, &ni, &ki,
                  alpha + test, NULL/*a*/, lda + test, NULL/*b*/, ldb + test, beta + test, NULL/*c*/, ldc + test);
                fprintf(stderr, "\n");
# endif
                result = EXIT_FAILURE;
              }
            }
          }
          /* avoid drift between Gold and test-results in case of beta!=0 */
          if (LIBXSMM_NEQ(0, beta[test])) {
            memcpy(c, gold, sizeof(OTYPE) * max_size_c);
            memcpy(d, gold, sizeof(OTYPE) * max_size_c);
          }
        }
      }
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
  libxsmm_free(d);
#if !defined(__BLAS) || (0 != __BLAS)
  libxsmm_free(gold);
#endif
  return result;
}

