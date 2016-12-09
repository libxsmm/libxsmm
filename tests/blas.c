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
#if !defined(NON_DEFAULT_REFERENCE_BLAS)
# define REFERENCE_BLAS LIBXSMM_BLAS_GEMM_SYMBOL
#else
# define REFERENCE_BLAS LIBXSMM_XBLAS_SYMBOL
#endif
#if !defined(NON_DEFAULT_LIBXSMM_BLAS)
# define LIBXSMM_BLAS LIBXSMM_XGEMM_SYMBOL
#else
# define LIBXSMM_BLAS LIBXSMM_YGEMM_SYMBOL
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


int main(void)
{
#if !defined(__BLAS) || (0 != __BLAS)
  const char transa = 'N', transb = 'N';
  libxsmm_blasint m[]     = { 1, 3,  64,    16, 350, 350, 350, 350, 350,  5 };
  libxsmm_blasint n[]     = { 1, 3, 239, 65792,  16,   1,  25,   4,   9, 13 };
  libxsmm_blasint k[]     = { 1, 3,  64,    16,  20,   1,  35,   4,  10, 70 };
  libxsmm_blasint lda[]   = { 1, 3,  64,    16, 350, 350, 350, 350, 350,  5 };
  libxsmm_blasint ldb[]   = { 1, 3, 240,    16,  35,  35,  35,  35,  35, 70 };
  libxsmm_blasint ldc[]   = { 1, 3, 240,    16, 350, 350, 350, 350, 350,  5 };
  const REAL_TYPE alpha[] = { 1, 1,   1,     1,   1,   1,   1,   1,   1,  1 };
  const REAL_TYPE beta[]  = { 1, 1,   1,     0,   0,   0,   0,   0,   0,  0 };
  const int start = 0, ntests = sizeof(m) / sizeof(*m);
  libxsmm_blasint maxm = 0, maxn = 0, maxk = 0, maxa = 0, maxb = 0, maxc = 0;
  REAL_TYPE *a = 0, *b = 0, *c = 0, *d = 0;
  int test, nwarnings = 0;
  libxsmm_blasint i, j;
  double d2 = 0;

  for (test = start; test < ntests; ++test) {
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

  init(42, a, maxm, maxk, maxa, 1.0);
  init(24, b, maxk, maxn, maxb, 1.0);
  init( 0, c, maxm, maxn, maxc, 1.0);
  init( 0, d, maxm, maxn, maxc, 1.0);

  for (test = start; test < ntests; ++test) {
    double dtest = 0;

    LIBXSMM_BLAS(REAL_TYPE)(&transa, &transb, m + test, n + test, k + test,
      alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);

    REFERENCE_BLAS(REAL_TYPE)(&transa, &transb, m + test, n + test, k + test,
      alpha + test, a, lda + test, b, ldb + test, beta + test, d, ldc + test);

    for (i = 0; i < n[test]; ++i) {
      for (j = 0; j < m[test]; ++j) {
        const libxsmm_blasint h = i * ldc[test] + j;
        const double d1 = c[h] - d[h];
        c[h] = d[h]; /* count error only once */
        dtest += d1 * d1;
      }
    }
    d2 = LIBXSMM_MAX(d2, dtest);
  }

  free(a); free(b);
  free(c); free(d);

  if (0.001 > d2) {
    return EXIT_SUCCESS;
  }
  else {
# if defined(_DEBUG)
    fprintf(stderr, "diff=%f\n", d2);
# endif
    return EXIT_FAILURE;
  }
#else
# if defined(_DEBUG)
  fprintf(stderr, "Warning: skipped the actual test due to missing BLAS support!\n");
# endif
  return EXIT_SUCCESS;
#endif
}

