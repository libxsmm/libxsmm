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
# define REAL_TYPE float
#endif
#if !defined(NTESTS)
# define NTESTS 10000
#endif
#if !defined(USE_DESCRIPTOR)
# define USE_DESCRIPTOR
#endif


int main(void)
{
  const int m[]           = {  1, 2, 3, LIBXSMM_MAX_M - 1, LIBXSMM_MAX_M, LIBXSMM_MAX_M + 1,    16,    16,    16,   32 };
  const int n[]           = {  1, 2, 3, LIBXSMM_MAX_N - 1, LIBXSMM_MAX_N, LIBXSMM_MAX_N + 1, 65279, 65280, 65792,   33 };
  const int k[]           = {  1, 2, 3, LIBXSMM_MAX_K - 1, LIBXSMM_MAX_K, LIBXSMM_MAX_K + 1,    16,    16,    16,  192 };
  libxsmm_blasint lda[]   = {  1, 2, 3, LIBXSMM_MAX_M - 1, LIBXSMM_MAX_M, LIBXSMM_MAX_M + 1,    16,    16,    16,   32 };
  libxsmm_blasint ldb[]   = {  1, 2, 3, LIBXSMM_MAX_K - 1, LIBXSMM_MAX_K, LIBXSMM_MAX_K + 1,    16,    16,    16, 2048 };
  libxsmm_blasint ldc[]   = {  1, 2, 3, LIBXSMM_MAX_M - 1, LIBXSMM_MAX_M, LIBXSMM_MAX_M + 1,    16,    16,    16, 2048 };
  const REAL_TYPE alpha[] = {  1, 1, 1,     LIBXSMM_ALPHA,             1,     LIBXSMM_ALPHA,     1,     1,     1,    1 };
  const REAL_TYPE beta[]  = {  1, 1, 1,      LIBXSMM_BETA,             0,      LIBXSMM_BETA,     0,     0,     0,    0 };
  const int size = sizeof(m) / sizeof(*m), flags = LIBXSMM_FLAGS, prefetch = LIBXSMM_PREFETCH_NONE;
  LIBXSMM_MMFUNCTION_TYPE(REAL_TYPE) f[sizeof(m)/sizeof(*m)];
  int i, nerrors = 0;

  /* initially generate a number of test kernels */
  for (i = 0; i < size; ++i) {
    f[i] = LIBXSMM_MMDISPATCH_SYMBOL(REAL_TYPE)(
      m[i], n[i], k[i], lda + i, ldb + i, ldc + i,
      alpha + i, beta + i, &flags, &prefetch);
  }

  /* check that the same kernels are dispatched as previously generated */
  for (i = 0; i < (NTESTS); ++i) {
#if defined(USE_DESCRIPTOR)
    libxsmm_xmmfunction fi = { 0 };
    LIBXSMM_GEMM_DESCRIPTOR_TYPE(descriptor, LIBXSMM_ALIGNMENT, flags | LIBXSMM_GEMM_TYPEFLAG(REAL_TYPE),
      m[i%size], n[i%size], k[i%size], lda[i%size], ldb[i%size], ldc[i%size],
      alpha[i%size], beta[i%size], prefetch);
    if (LIBXSMM_GEMM_NO_BYPASS_DIMS(m[i%size], n[i%size], k[i%size]) /* account for BIG=0 descriptor */
     && LIBXSMM_GEMM_NO_BYPASS_DIMS(lda[i%size], ldb[i%size], ldc[i%size]))
    {
      fi = libxsmm_xmmdispatch(&descriptor);
    }
    if (fi.LIBXSMM_TPREFIX(REAL_TYPE,mm) != f[i%size])
#else
    const LIBXSMM_MMFUNCTION_TYPE(REAL_TYPE) fi = LIBXSMM_MMDISPATCH_SYMBOL(REAL_TYPE)(
      m[i%size], n[i%size], k[i%size], lda + (i % size), ldb + (i % size), ldc + (i % size),
      alpha + (i % size), beta + (i % size), &flags, &prefetch);
    if (fi != f[i%size])
#endif
    { /* always an error even when JIT is disabled at compile-time */
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

