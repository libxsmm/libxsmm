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
#include <stdio.h>

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif

/*#define USE_LIBXSMM_BLAS*/
#define M 64
#define N 239
#define K 64
#define LDA 64
#define LDB 240
#define LDC 240


int main(void)
{
  const libxsmm_blasint m = M, n = N, k = K, lda = LDA, ldb = LDB, ldc = LDC;
  REAL_TYPE a[K*LDA], b[N*LDB], c[N*LDC], d[N*LDC];
  const REAL_TYPE alpha = 1, beta = 1;
  const char notrans = 'N';
  double d2 = 0;
  int i, j;

  for (i = 0; i < m; ++i) {
    for (j = 0; j < k; ++j) {
      const int index = i * lda + j;
      a[index] = ((REAL_TYPE)1) / (index + 1);
    }
  }
  for (i = 0; i < k; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldb + j;
      b[index] = ((REAL_TYPE)2) / (index + 1);
    }
  }
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldc + j;
      c[index] = d[index] = 1000;
    }
  }

  LIBXSMM_XGEMM_SYMBOL(REAL_TYPE)(&notrans, &notrans, &m, &n, &k,
    &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

#if defined(USE_LIBXSMM_BLAS)
  LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&notrans, &notrans, &m, &n, &k,
    &alpha, a, &lda, b, &ldb, &beta, d, &ldc);
#else
  LIBXSMM_BLAS_GEMM_SYMBOL(REAL_TYPE)(&notrans, &notrans, &m, &n, &k,
    &alpha, a, &lda, b, &ldb, &beta, d, &ldc);
#endif

  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldc + j;
      const double d1 = c[index] - d[index];
      d2 += d1 * d1;
    }
  }

  return 0.001 > d2 ? EXIT_SUCCESS : EXIT_FAILURE;
}

