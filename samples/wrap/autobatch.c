/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
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
#include <stdio.h>

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif
#if !defined(DGEMM)
# define DGEMM LIBXSMM_FSYMBOL(LIBXSMM_CONCATENATE(__wrap_, LIBXSMM_TPREFIX(REAL_TYPE, gemm)))
#endif


/** Function prototype for LIBXSMM's wrapped DGEMM; this way auto-batch can be tested as if DGEMM calls are intercepted/batched. */
void DGEMM(const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const REAL_TYPE*, const REAL_TYPE*, const libxsmm_blasint*, const REAL_TYPE*, const libxsmm_blasint*,
  const REAL_TYPE*, REAL_TYPE*, const libxsmm_blasint*);


void init(int seed, REAL_TYPE* dst, libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld, double scale);
void init(int seed, REAL_TYPE* dst, libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld, double scale)
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


int main(int argc, char* argv[])
{
  const libxsmm_blasint m = 2 < argc ? atoi(argv[1]) : 23;
  const libxsmm_blasint k = 3 < argc ? atoi(argv[3]) : m;
  const libxsmm_blasint n = 2 < argc ? atoi(argv[2]) : k;
  const libxsmm_blasint size = 4 < argc ? atoi(argv[4]) : 1000;
  const libxsmm_blasint lda = m, ldb = k, ldc = m;
  const REAL_TYPE alpha = 1.0, beta = 1.0;
  const char transa = 'N', transb = 'N';
  const int flags = LIBXSMM_GEMM_FLAGS(transa, transb) /*| LIBXSMM_MMBATCH_FLAG_SEQUENTIAL*/;
  REAL_TYPE *a = 0, *b = 0, *c = 0;
  int result = EXIT_SUCCESS, i;

  a = (REAL_TYPE*)malloc(lda * k * sizeof(REAL_TYPE));
  b = (REAL_TYPE*)malloc(ldb * n * sizeof(REAL_TYPE));
  c = (REAL_TYPE*)malloc(ldc * n * sizeof(REAL_TYPE));
  if (0 == a || 0 == b || 0 == c) {
    result = EXIT_FAILURE;
  }

  libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(REAL_TYPE),
    &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

  if (EXIT_SUCCESS == result) {
    init(42, a, m, k, lda, 1.0);
    init(24, b, k, n, ldb, 1.0);
    init(0, c, m, n, ldc, 1.0);

    /* enable batch-recording of the specified matrix multiplication */
    libxsmm_mmbatch_begin(LIBXSMM_GEMM_PRECISION(REAL_TYPE), &flags, &m, &n, &k, &lda, &ldb, &ldc, &alpha, &beta);

    for (i = 0; i < size; ++i) {
      DGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    /* disable/flush multiplication batch */
    result = libxsmm_mmbatch_end();
  }

  free(a);
  free(b);
  free(c);

  return result;
}

