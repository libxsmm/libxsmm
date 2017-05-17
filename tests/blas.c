/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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
  libxsmm_blasint m[]     = { 0, 0, 1, 1, 3, 3, 1,  64,    16,    16, 350, 350, 350, 350, 350,  5, 10, 12, 20,   32 };
  libxsmm_blasint n[]     = { 0, 1, 1, 1, 3, 1, 3, 239, 13824, 65792,  16,   1,  25,   4,   9, 13,  1, 10,  6,   33 };
  libxsmm_blasint k[]     = { 0, 1, 1, 1, 3, 2, 2,  64,    16,    16,  20,   1,  35,   4,  10, 70,  1, 12,  6,  192 };
  libxsmm_blasint lda[]   = { 0, 1, 1, 1, 3, 3, 1,  64,    16,    16, 350, 350, 350, 350, 350,  5, 22, 22, 22,   32 };
  libxsmm_blasint ldb[]   = { 0, 1, 1, 1, 3, 2, 2, 240,    16,    16,  35,  35,  35,  35,  35, 70,  1, 20,  8, 2048 };
  libxsmm_blasint ldc[]   = { 0, 1, 0, 1, 3, 3, 1, 240,    16,    16, 350, 350, 350, 350, 350,  5, 22, 12, 20, 2048 };
  const REAL_TYPE alpha[] = { 1, 1, 1, 1, 1, 1, 1,   1,     1,     1,   1,   1,   1,   1,   1,  1,  1,  1,  1,    1 };
  const REAL_TYPE beta[]  = { 0, 1, 0, 1, 1, 0, 0,   1,     0,     0,   0,   0,   0,   0,   0,  0,  0,  0,  0,    0 };
  const int begin = 3, end = sizeof(m) / sizeof(*m);
  libxsmm_blasint maxm = 1, maxn = 1, maxk = 1, maxa = 1, maxb = 1, maxc = 1;
  REAL_TYPE *a = 0, *b = 0, *c = 0, *d = 0;
  libxsmm_blasint i, j;
  double d2 = 0;
  int test;

  for (test = begin; test < end; ++test) {
    maxm = LIBXSMM_MAX(maxm, m[test]);
    maxn = LIBXSMM_MAX(maxn, n[test]);
    maxk = LIBXSMM_MAX(maxk, k[test]);
    maxa = LIBXSMM_MAX(maxa, lda[test]);
    maxb = LIBXSMM_MAX(maxb, ldb[test]);
    maxc = LIBXSMM_MAX(maxc, ldc[test]);
  }

  a = (REAL_TYPE*)libxsmm_malloc(maxa * maxk * sizeof(REAL_TYPE));
  b = (REAL_TYPE*)libxsmm_malloc(maxb * maxn * sizeof(REAL_TYPE));
  c = (REAL_TYPE*)libxsmm_malloc(maxc * maxn * sizeof(REAL_TYPE));
  d = (REAL_TYPE*)libxsmm_malloc(maxc * maxn * sizeof(REAL_TYPE));
  assert(0 != a && 0 != b && 0 != c && 0 != d);

  init(42, a, maxm, maxk, maxa, 1.0);
  init(24, b, maxk, maxn, maxb, 1.0);
  init( 0, c, maxm, maxn, maxc, 1.0);
  init( 0, d, maxm, maxn, maxc, 1.0);

  for (test = begin; test < end; ++test) {
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

  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);
  libxsmm_free(d);

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
  fprintf(stderr, "Warning: skipped the test due to missing BLAS support!\n");
# endif
  return EXIT_SUCCESS;
#endif
}

