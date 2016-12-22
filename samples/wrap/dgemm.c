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
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#if !defined(BLASINT_TYPE)
# define BLASINT_TYPE int
#endif

/** Function prototype for DGEMM; this way any kind of LAPACK/BLAS library is sufficient at link-time. */
void dgemm_(const char*, const char*, const BLASINT_TYPE*, const BLASINT_TYPE*, const BLASINT_TYPE*,
  const double*, const double*, const BLASINT_TYPE*, const double*, const BLASINT_TYPE*,
  const double*, double*, const BLASINT_TYPE*);


void init(int seed, double* dst, BLASINT_TYPE nrows, BLASINT_TYPE ncols, BLASINT_TYPE ld, double scale);
void init(int seed, double* dst, BLASINT_TYPE nrows, BLASINT_TYPE ncols, BLASINT_TYPE ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  BLASINT_TYPE i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    BLASINT_TYPE j = 0;
    for (; j < nrows; ++j) {
      const BLASINT_TYPE k = i * ld + j;
      dst[k] = (double)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const BLASINT_TYPE k = i * ld + j;
      dst[k] = (double)seed;
    }
  }
}


int main(int argc, char* argv[])
{
  int size = 2 == argc ? atoi(argv[1]) : 500;
  const BLASINT_TYPE m = 2 < argc ? atoi(argv[1]) : 23;
  const BLASINT_TYPE n = 2 < argc ? atoi(argv[2]) : m;
  const BLASINT_TYPE k = 3 < argc ? atoi(argv[3]) : m;
  const BLASINT_TYPE lda = 4 < argc ? atoi(argv[4]) : m;
  const BLASINT_TYPE ldb = 5 < argc ? atoi(argv[5]) : k;
  const BLASINT_TYPE ldc = 6 < argc ? atoi(argv[6]) : m;
  const double alpha = 7 < argc ? atof(argv[7]) : 1.0;
  const double beta = 8 < argc ? atof(argv[8]) : 1.0;
  const char transa = 'N', transb = 'N';
  double *a = 0, *b = 0, *c = 0;
  int i;

  if (7 < argc) size = atoi(argv[7]);
  a = (double*)malloc(lda * k * sizeof(double));
  b = (double*)malloc(ldb * n * sizeof(double));
  c = (double*)malloc(ldc * n * sizeof(double));
  printf("dgemm('%c', '%c', %i/*m*/, %i/*n*/, %i/*k*/,\n"
         "      %g/*alpha*/, %p/*a*/, %i/*lda*/,\n"
         "                  %p/*b*/, %i/*ldb*/,\n"
         "       %g/*beta*/, %p/*c*/, %i/*ldc*/)\n",
    transa, transb, m, n, k, alpha, (const void*)a, lda,
                                    (const void*)b, ldb,
                              beta, (const void*)c, ldc);

  assert(0 != a && 0 != b && 0 != c);
  init(42, a, m, k, lda, 1.0);
  init(24, b, k, n, ldb, 1.0);
  init( 0, c, m, n, ldc, 1.0);

  for (i = 0; i < size; ++i) {
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  printf("Called %i times.\n", size);

  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}

