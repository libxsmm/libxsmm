/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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

#if LIBXSMM_EQUAL(ELEM_TYPE, float) || LIBXSMM_EQUAL(ELEM_TYPE, double)
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
#   include <mkl_trans.h>
#   define MATCOPY_GOLD(M, N, A, LDI, B, LDO) \
      LIBXSMM_CONCATENATE(mkl_, LIBXSMM_TPREFIX(ELEM_TYPE, omatcopy))('C', 'n', \
        (size_t)(*(M)), (size_t)(*(N)), (ELEM_TYPE)1, A, (size_t)(*(LDI)), B, (size_t)(*(LDO)))
# elif defined(__OPENBLAS77)
#   include <f77blas.h>
#   define MATCOPY_GOLD(M, N, A, LDI, B, LDO) { \
      /*const*/char matcopy_gold_tc_ = 'C', matcopy_gold_tt_ = 'n'; \
      /*const*/ELEM_TYPE matcopy_gold_alpha_ = 1; \
      LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(ELEM_TYPE, omatcopy))(&matcopy_gold_tc_, &matcopy_gold_tt_, \
        (libxsmm_blasint*)(M), (libxsmm_blasint*)(N), &matcopy_gold_alpha_, \
        A, (libxsmm_blasint*)(LDI), B, (libxsmm_blasint*)(LDO)); \
    }
# endif
#endif

#if !defined(CHECK_PARALLEL)
# define CHECK_PARALLEL
#endif

#if defined(CHECK_PARALLEL)
# define MATCOPY libxsmm_matcopy_omp
#else
# define MATCOPY libxsmm_matcopy
#endif


int main(void)
{
  const libxsmm_blasint m[]   = { 1, 1, 1, 1, 2, 3, 6, 6, 6, 6,   8, 16, 63,  16,  16, 2507 };
  const libxsmm_blasint n[]   = { 1, 6, 7, 7, 2, 3, 1, 1, 1, 1, 250, 16, 31, 500, 500, 1975 };
  const libxsmm_blasint ldi[] = { 1, 1, 2, 2, 2, 3, 6, 8, 6, 7, 512, 16, 63,  16, 512, 3000 };
  const libxsmm_blasint ldo[] = { 1, 1, 1, 8, 2, 3, 6, 6, 8, 8,  16, 16, 64, 512,  16, 3072 };
  const int prefetch[]        = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,   0,  1,  0,   1,   0,    1 };
  const int start = 0, ntests = sizeof(m) / sizeof(*m);
  libxsmm_blasint max_size_a = 0, max_size_b = 0;
  unsigned int nerrors = 0;
  ELEM_TYPE *a = 0, *b = 0;
#if defined(MATCOPY_GOLD)
  ELEM_TYPE *c = 0;
#endif
  int test;

  for (test = start; test < ntests; ++test) {
    const libxsmm_blasint size_a = ldi[test] * n[test], size_b = ldo[test] * n[test];
    assert(m[test] <= ldi[test] && m[test] <= ldo[test]);
    max_size_a = LIBXSMM_MAX(max_size_a, size_a);
    max_size_b = LIBXSMM_MAX(max_size_b, size_b);
  }
  a = (ELEM_TYPE*)libxsmm_malloc((size_t)(max_size_a * sizeof(ELEM_TYPE)));
  b = (ELEM_TYPE*)libxsmm_malloc((size_t)(max_size_b * sizeof(ELEM_TYPE)));
  assert(0 != a && 0 != b);

  LIBXSMM_MATINIT(ELEM_TYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
  LIBXSMM_MATINIT(ELEM_TYPE,  0, b, max_size_b, 1, max_size_b, 1.0);
#if defined(MATCOPY_GOLD)
  c = (ELEM_TYPE*)libxsmm_malloc((size_t)(max_size_b * sizeof(ELEM_TYPE)));
  assert(0 != c);
  LIBXSMM_MATINIT(ELEM_TYPE, 0, c, max_size_b, 1, max_size_b, 1.0);
#endif
  for (test = start; test < ntests; ++test) {
    MATCOPY(b, a, sizeof(ELEM_TYPE), m[test], n[test], ldi[test], ldo[test], prefetch + test);
    { /* validation */
      unsigned int testerrors = 0;
      libxsmm_blasint i, j;
      for (i = 0; i < n[test]; ++i) {
        for (j = 0; j < m[test]; ++j) {
          const ELEM_TYPE u = a[i*ldi[test]+j];
          const ELEM_TYPE v = b[i*ldo[test]+j];
          if (LIBXSMM_NEQ(u, v)) {
            ++testerrors;
          }
        }
      }
#if defined(MATCOPY_GOLD)
      if (0 == testerrors) {
        MATCOPY_GOLD(m + test, n + test, a, ldi + test, c, ldo + test);
        for (i = 0; i < n[test]; ++i) {
          for (j = 0; j < m[test]; ++j) {
            const ELEM_TYPE u = b[i*ldo[test] + j];
            const ELEM_TYPE v = c[i*ldo[test] + j];
            if (LIBXSMM_NEQ(u, v)) {
              ++testerrors;
            }
          }
        }
      }
#endif
      if (nerrors < testerrors) {
        nerrors = testerrors;
      }
    }
  }

  libxsmm_free(a);
  libxsmm_free(b);
#if defined(MATCOPY_GOLD)
  libxsmm_free(c);
#endif

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

