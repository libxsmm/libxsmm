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

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif
#if !defined(NTESTS)
# define NTESTS 10000
#endif


int main(void)
{
  const int m[] = { 1, 2, 3, LIBXSMM_MAX_M - 1, LIBXSMM_MAX_M, LIBXSMM_MAX_M + 1,    16,    16,    16 };
  const int n[] = { 1, 2, 3, LIBXSMM_MAX_N - 1, LIBXSMM_MAX_N, LIBXSMM_MAX_N + 1, 65279, 65280, 65792 };
  const int k[] = { 1, 2, 3, LIBXSMM_MAX_K - 1, LIBXSMM_MAX_K, LIBXSMM_MAX_K + 1,    16,    16,    16 };
  const int size = sizeof(m) / sizeof(*m), flags = LIBXSMM_FLAGS, prefetch = LIBXSMM_PREFETCH;
  const REAL_TYPE alpha = LIBXSMM_ALPHA, beta = LIBXSMM_BETA;
  LIBXSMM_MMFUNCTION_TYPE(REAL_TYPE) f[sizeof(m)/sizeof(*m)];
  int i, nerrors = 0;

  /* initially generate a number of test kernels */
  for (i = 0; i < size; ++i) {
    f[i] = LIBXSMM_MMDISPATCH_SYMBOL(REAL_TYPE)(
      m[i], n[i], k[i], m + i, k + i, m + i,
      &alpha, &beta, &flags, &prefetch);
  }

  /* check that the same kernels are dispatched as previously generated */
  for (i = 0; i < (NTESTS); ++i) {
    const LIBXSMM_MMFUNCTION_TYPE(REAL_TYPE) fi = LIBXSMM_MMDISPATCH_SYMBOL(REAL_TYPE)(
      m[i%size], n[i%size], k[i%size], m + (i % size), k + (i % size), m + (i % size),
      &alpha, &beta, &flags, &prefetch);

    if (fi != f[i%size]) { /* always an error even when JIT is disabled at compile-time */
#if defined(_DEBUG)
      if (0 != fi) {
        fprintf(stderr, "Error: the %ix%ix%i-kernel does not match!\n", m[i%size], n[i%size], k[i%size]);
      }
      else { /* did not find previously generated and recorded kernel */
        fprintf(stderr, "Error: cannot find %ix%ix%i-kernel!\n", m[i%size], n[i%size], k[i%size]);
      }
#endif
      ++nerrors;
    }
  }

  return (0 == nerrors) ? EXIT_SUCCESS : EXIT_FAILURE;
}

