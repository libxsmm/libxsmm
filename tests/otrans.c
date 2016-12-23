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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE float
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
  libxsmm_blasint m[]     = { 1, 2, 3, 16, 63,  16 };
  libxsmm_blasint n[]     = { 1, 2, 3, 16, 31, 500 };
  libxsmm_blasint ldi[]   = { 1, 2, 3, 16, 64,  16 };
  libxsmm_blasint ldo[]   = { 1, 2, 3, 16, 32, 512 };
  const int start = 0, ntests = sizeof(m) / sizeof(*m);
  libxsmm_blasint maxm = 0, maxn = 0, maxi = 0, maxo = 0, nerrors = 0;
  REAL_TYPE *a = 0, *b = 0;
  libxsmm_blasint i, j;
  int test;

  for (test = start; test < ntests; ++test) {
    maxm = LIBXSMM_MAX(maxm, m[test]);
    maxn = LIBXSMM_MAX(maxn, n[test]);
    maxi = LIBXSMM_MAX(maxi, ldi[test]);
    maxo = LIBXSMM_MAX(maxo, ldo[test]);
  }
  a = (REAL_TYPE*)libxsmm_malloc(maxi * maxn * sizeof(REAL_TYPE));
  b = (REAL_TYPE*)libxsmm_malloc(maxo * maxm * sizeof(REAL_TYPE));
  assert(0 != a && 0 != b);

  init(42, a, maxm, maxn, maxi, 1.0);
  init( 0, b, maxn, maxm, maxo, 1.0);

  for (test = start; test < ntests; ++test) {
    libxsmm_blasint testerrors = (EXIT_SUCCESS == libxsmm_otrans(
      b, a, sizeof(REAL_TYPE), m[test], n[test],
      ldi[test], ldo[test]) ? 0 : 1);

    if (0 == testerrors) {
      for (i = 0; i < n[test]; ++i) {
        for (j = i + 1; j < m[test]; ++j) {
          const libxsmm_blasint u = i * ldi[test] + j;
          const libxsmm_blasint v = j * ldo[test] + i;
          testerrors += (LIBXSMM_FEQ(a[u], b[v]) ? 0 : 1);
        }
      }
    }
    nerrors = LIBXSMM_MAX(nerrors, testerrors);
  }

  if (0 == nerrors) { /* previous results are correct and may be used to validate other tests */
    for (test = start; test < ntests; ++test) {
      /* prepare expected results in b (correct according to the previous test block) */
      libxsmm_otrans(b, a, sizeof(REAL_TYPE), m[test], n[test], ldi[test], ldo[test]);

      if (m[test] == n[test] && ldi[test] == ldo[test]) {
        libxsmm_blasint testerrors = (EXIT_SUCCESS == libxsmm_otrans(
          a, a, sizeof(REAL_TYPE), m[test], n[test],
          ldi[test], ldo[test]) ? 0 : 1);

        if (0 == testerrors) {
          for (i = 0; i < n[test]; ++i) {
            for (j = i + 1; j < m[test]; ++j) {
              /* address serves both a and b since ldi and ldo are equal */
              const libxsmm_blasint uv = i * ldi[test] + j;
              testerrors += (LIBXSMM_FEQ(a[uv], b[uv]) ? 0 : 1);
            }
          }
        }
        nerrors = LIBXSMM_MAX(nerrors, testerrors);
      }
      else { /* negative tests */
        nerrors = LIBXSMM_MAX(EXIT_SUCCESS != libxsmm_otrans(
          a, a, sizeof(REAL_TYPE), m[test], n[test],
          ldi[test], ldo[test]) ? 0 : 1, nerrors);
      }
    }
  }

  libxsmm_free(a);
  libxsmm_free(b);

  if (0 == nerrors) {
    return EXIT_SUCCESS;
  }
  else {
# if defined(_DEBUG)
    fprintf(stderr, "errors=%i\n", nerrors);
# endif
    return EXIT_FAILURE;
  }
}

