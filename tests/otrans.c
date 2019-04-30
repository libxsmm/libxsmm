/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
#include <string.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif

#if !defined(ELEM_TYPE)
# define ELEM_TYPE double
#endif

#if !defined(CHECK_PARALLEL)
# define CHECK_PARALLEL
#endif

#if defined(CHECK_PARALLEL)
# define OTRANS libxsmm_otrans_omp
#else
# define OTRANS libxsmm_otrans
#endif


int main(void)
{
  /* test #:                      1  2  3  4  5  6  7  8  9 10 11  12  13  14  15  16  17  18   19   20   21    22 */
  const libxsmm_blasint m[]   = { 0, 1, 1, 1, 1, 2, 3, 5, 5, 5, 5,  5, 13, 13, 16, 22, 63, 64,  16,  16,  75, 2507 };
  const libxsmm_blasint n[]   = { 0, 1, 7, 7, 7, 2, 3, 1, 1, 1, 5, 13,  5, 13, 16, 22, 31, 64, 500,  32, 130, 1975 };
  const libxsmm_blasint ldi[] = { 0, 1, 1, 1, 9, 2, 3, 5, 8, 8, 5,  5, 13, 13, 16, 22, 64, 64,  16, 512,  87, 3000 };
  const libxsmm_blasint ldo[] = { 1, 1, 7, 8, 8, 2, 3, 1, 1, 4, 5, 13,  5, 13, 16, 22, 32, 64, 512,  64, 136, 3072 };
  const int start = 0, ntests = sizeof(m) / sizeof(*m);
  libxsmm_blasint max_size_a = 0, max_size_b = 0;
  ELEM_TYPE *a = NULL, *b = NULL, *c = NULL;
  unsigned int nerrors = 0;
  int test;

  for (test = start; test < ntests; ++test) {
    const libxsmm_blasint size_a = ldi[test] * n[test], size_b = ldo[test] * m[test];
    LIBXSMM_ASSERT(m[test] <= ldi[test] && n[test] <= ldo[test]);
    max_size_a = LIBXSMM_MAX(max_size_a, size_a);
    max_size_b = LIBXSMM_MAX(max_size_b, size_b);
  }
  a = (ELEM_TYPE*)libxsmm_malloc((size_t)(sizeof(ELEM_TYPE) * max_size_a));
  b = (ELEM_TYPE*)libxsmm_malloc((size_t)(sizeof(ELEM_TYPE) * max_size_b));
  c = (ELEM_TYPE*)libxsmm_malloc((size_t)(sizeof(ELEM_TYPE) * max_size_b));
  LIBXSMM_ASSERT(NULL != a && NULL != b && NULL != c);

  /* initialize data */
  LIBXSMM_MATINIT(ELEM_TYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
  LIBXSMM_MATINIT(ELEM_TYPE, 24, b, max_size_b, 1, max_size_b, 1.0);

  for (test = start; test < ntests; ++test) {
    OTRANS(b, a, sizeof(ELEM_TYPE), m[test], n[test], ldi[test], ldo[test]);
    { /* validation */
      unsigned int testerrors = 0;
      libxsmm_blasint i, j;
      memcpy(c, b, (size_t)(sizeof(ELEM_TYPE) * max_size_b));
      for (i = 0; i < n[test]; ++i) {
        for (j = 0; j < m[test]; ++j) {
          const libxsmm_blasint u = i * ldi[test] + j;
          const libxsmm_blasint v = j * ldo[test] + i;
          testerrors += LIBXSMM_NEQ(a[u], b[v]);
        }
        for (j = m[test]; j < ldi[test]; ++j) {
          const libxsmm_blasint v = j * ldo[test] + i;
          if (v < max_size_b) {
            testerrors += LIBXSMM_NEQ(b[v], c[v]);
          }
        }
      }
      for (i = n[test]; i < ldo[test]; ++i) {
        for (j = 0; j < m[test]; ++j) {
          const libxsmm_blasint v = j * ldo[test] + i;
          if (v < max_size_b) {
            testerrors += LIBXSMM_NEQ(b[v], c[v]);
          }
          else {
            ++testerrors;
          }
        }
        for (j = m[test]; j < ldi[test]; ++j) {
          const libxsmm_blasint v = j * ldo[test] + i;
          if (v < max_size_b) {
            testerrors += LIBXSMM_NEQ(b[v], c[v]);
          }
        }
      }
      if (nerrors < testerrors) {
        nerrors = testerrors;
      }
    }
  }

  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);

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

