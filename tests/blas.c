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
#include <stdlib.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif

#define MAXREDUCE(I, VALUE, MAX, LIMIT, NWARNINGS) \
  if ((LIMIT) < ((VALUE)[I])) { \
    (VALUE)[I] = LIMIT; \
    ++(NWARNINGS); \
  } \
  MAX = LIBXSMM_MAX(MAX, (VALUE)[I])

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif

/*#define USE_LIBXSMM_BLAS*/


int main(void)
{
  const char transa = 'N', transb = 'N';
  libxsmm_blasint m[]     = { 1, 3,  64,    16,  30,  30,  30,  30,  30,  30,  30,  30,  30,  5,  64,  64,  64,  64,  64,  64,  64,  64,  64 };
  libxsmm_blasint n[]     = { 1, 3, 239, 65792,   1,   1,   1,   1,   4,   8,   8,   8,   8, 13,   1,   1,   1,   1,   4,   8,   8,   8,   8 };
  libxsmm_blasint k[]     = { 1, 3,  64,    16,   1,   2,   3,   8,   4,   2,   3,   4,   8, 70,   1,   2,   3,   8,   4,   2,   3,   4,   8 };
  libxsmm_blasint lda[]   = { 1, 3,  64,    16, 350, 350, 350, 350, 350, 350, 350, 350, 350,  5, 350, 350, 350, 350, 350, 350, 350, 350, 350 };
  libxsmm_blasint ldb[]   = { 1, 3, 240,    16,  35,  35,  35,  35,  35,  35,  35,  35,  35, 70,  35,  35,  35,  35,  35,  35,  35,  35,  35 };
  libxsmm_blasint ldc[]   = { 1, 3, 240,    16, 350, 350, 350, 350, 350, 350, 350, 350, 350,  5, 350, 350, 350, 350, 350, 350, 350, 350, 350 };
  const REAL_TYPE alpha[] = { 1, 1,   1,     1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1 };
  const REAL_TYPE beta[]  = { 1, 1,   1,     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0 };
  const int ntests = sizeof(m) / sizeof(*m);
  libxsmm_blasint maxm = 0, maxn = 0, maxk = 0, maxa = 0, maxb = 0, maxc = 0;
  REAL_TYPE *a = 0, *b = 0, *c = 0, *d = 0;
  int test, nwarnings = 0;
  libxsmm_blasint i, j;
  double d2 = 0;

  for (test = 0; test < ntests; ++test) {
    MAXREDUCE(test, m, maxm, LIBXSMM_GEMM_DESCRIPTOR_DIM_MAX, nwarnings);
    MAXREDUCE(test, n, maxn, LIBXSMM_GEMM_DESCRIPTOR_DIM_MAX, nwarnings);
    MAXREDUCE(test, k, maxk, LIBXSMM_GEMM_DESCRIPTOR_DIM_MAX, nwarnings);
    MAXREDUCE(test, lda, maxa, LIBXSMM_GEMM_DESCRIPTOR_DIM_MAX, nwarnings);
    MAXREDUCE(test, ldb, maxb, LIBXSMM_GEMM_DESCRIPTOR_DIM_MAX, nwarnings);
    MAXREDUCE(test, ldc, maxc, LIBXSMM_GEMM_DESCRIPTOR_DIM_MAX, nwarnings);
  }
#if defined(_DEBUG)
  if (0 < nwarnings) {
    fprintf(stderr, "Warning: recompile with BIG=1 and an increased THRESHOLD for a complete test!\n");
  }
#endif

  a = (REAL_TYPE*)malloc(maxa * maxk * sizeof(REAL_TYPE));
  b = (REAL_TYPE*)malloc(maxb * maxn * sizeof(REAL_TYPE));
  c = (REAL_TYPE*)malloc(maxc * maxn * sizeof(REAL_TYPE));
  d = (REAL_TYPE*)malloc(maxc * maxn * sizeof(REAL_TYPE));
  assert(0 != a && 0 != b && 0 != c && 0 != d);

  for (j = 0; j < maxk; ++j) {
    for (i = 0; i < maxm; ++i) {
      const libxsmm_blasint index = j * maxa + i;
      a[index] = ((REAL_TYPE)1) / (index + 1);
    }
  }
  for (j = 0; j < maxn; ++j) {
    for (i = 0; i < maxk; ++i) {
      const libxsmm_blasint index = j * maxb + i;
      b[index] = ((REAL_TYPE)2) / (index + 1);
    }
  }
  for (j = 0; j < maxn; ++j) {
    for (i = 0; i < maxm; ++i) {
      const libxsmm_blasint index = j * maxc + i;
      c[index] = d[index] = 1000;
    }
  }

  for (test = 0; test < ntests; ++test) {
    double dtest = 0;
#if !defined(__BLAS) || (0 != __BLAS)
    LIBXSMM_XGEMM_SYMBOL(REAL_TYPE)(&transa, &transb, m + test, n + test, k + test,
      alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);

# if defined(USE_LIBXSMM_BLAS)
    LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, m + test, n + test, k + test,
      alpha + test, a, lda + test, b, ldb + test, beta + test, d, ldc + test);
# else
    LIBXSMM_BLAS_GEMM_SYMBOL(REAL_TYPE)(&transa, &transb, m + test, n + test, k + test,
      alpha + test, a, lda + test, b, ldb + test, beta + test, d, ldc + test);
# endif

    for (j = 0; j < n[test]; ++j) {
      for (i = 0; i < m[test]; ++i) {
        const libxsmm_blasint index = j * ldc[test] + i;
        const double d1 = c[index] - d[index];
        c[index] = d[index];
        dtest += d1 * d1;
      }
    }
    d2 = LIBXSMM_MAX(d2, dtest);
  }

  free(a); free(b);
  free(c); free(d);

  return 0.001 > d2 ? EXIT_SUCCESS : EXIT_FAILURE;
#else
  LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(b); LIBXSMM_UNUSED(c);
  LIBXSMM_UNUSED(transa); LIBXSMM_UNUSED(transb);
  LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(beta);
  LIBXSMM_UNUSED(d2);
# if defined(_DEBUG)
  fprintf(stderr, "Warning: skipped the actual test due to missing BLAS support!\n");
# endif
  return EXIT_SUCCESS;
#endif
}

