/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
#include <assert.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif

#if !defined(ELEM_TYPE)
# define ELEM_TYPE float
#endif


int main(void)
{
  const libxsmm_blasint m[]   = { 1, 2, 3, 16, 63,  16,  75, 2507 };
  const libxsmm_blasint n[]   = { 1, 2, 3, 16, 31, 500, 130, 1975 };
  const libxsmm_blasint ldi[] = { 1, 2, 3, 16, 64,  16,  87, 3000 };
  const libxsmm_blasint ldo[] = { 1, 2, 3, 16, 32, 512, 136, 3072 };
  const int start = 0, ntests = sizeof(m) / sizeof(*m);
  libxsmm_blasint max_size_a = 0, max_size_b = 0;
  unsigned int nerrors = 0;
  ELEM_TYPE *a = 0, *b = 0;
  int test;

  for (test = start; test < ntests; ++test) {
    const libxsmm_blasint size_a = ldi[test] * n[test], size_b = ldo[test] * m[test];
    assert(m[test] <= ldi[test] && n[test] <= ldo[test]);
    max_size_a = LIBXSMM_MAX(max_size_a, size_a);
    max_size_b = LIBXSMM_MAX(max_size_b, size_b);
  }
  a = (ELEM_TYPE*)libxsmm_malloc((size_t)(max_size_a * sizeof(ELEM_TYPE)));
  b = (ELEM_TYPE*)libxsmm_malloc((size_t)(max_size_b * sizeof(ELEM_TYPE)));
  assert(0 != a && 0 != b);

  LIBXSMM_MATINIT(ELEM_TYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
  LIBXSMM_MATINIT(ELEM_TYPE,  0, b, max_size_b, 1, max_size_b, 1.0);

  for (test = start; test < ntests; ++test) {
    unsigned int testerrors = (EXIT_SUCCESS == libxsmm_otrans(
      b, a, sizeof(ELEM_TYPE), m[test], n[test],
      ldi[test], ldo[test]) ? 0 : 1);

    if (0 == testerrors) {
      libxsmm_blasint i, j;
      for (i = 0; i < n[test]; ++i) {
        for (j = 0; j < m[test]; ++j) {
          const libxsmm_blasint u = i * ldi[test] + j;
          const libxsmm_blasint v = j * ldo[test] + i;
          testerrors += (LIBXSMM_FEQ(a[u], b[v]) ? 0u : 1u);
        }
      }
    }
    nerrors = LIBXSMM_MAX(nerrors, testerrors);
  }

  if (0 == nerrors) { /* previous results are correct and may be used to validate other tests */
    for (test = start; test < ntests; ++test) {
      /* prepare expected results in b (correct according to the previous test block) */
      libxsmm_otrans(b, a, sizeof(ELEM_TYPE), m[test], n[test], ldi[test], ldo[test]);

      if (m[test] == n[test] && ldi[test] == ldo[test]) {
        unsigned int testerrors = (EXIT_SUCCESS == libxsmm_otrans(
          a, a, sizeof(ELEM_TYPE), m[test], n[test],
          ldi[test], ldo[test]) ? 0 : 1);

        if (0 == testerrors) {
          libxsmm_blasint i, j;
          for (i = 0; i < n[test]; ++i) {
            for (j = 0; j < m[test]; ++j) {
              /* address serves both a and b since ldi and ldo are equal */
              const libxsmm_blasint uv = i * ldi[test] + j;
              testerrors += (LIBXSMM_FEQ(a[uv], b[uv]) ? 0u : 1u);
            }
          }
        }
        nerrors = LIBXSMM_MAX(nerrors, testerrors);
      }
      else { /* negative tests */
        nerrors = LIBXSMM_MAX(EXIT_SUCCESS != libxsmm_otrans(
          a, a, sizeof(ELEM_TYPE), m[test], n[test],
          ldi[test], ldo[test]) ? 0u : 1u, nerrors);
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
    fprintf(stderr, "errors=%u\n", nerrors);
# endif
    return EXIT_FAILURE;
  }
}

