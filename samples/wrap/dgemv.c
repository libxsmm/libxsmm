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
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#if !defined(BLASINT_TYPE)
# define BLASINT_TYPE int
#endif

/** Function prototype for DGEMM; this way any kind of LAPACK/BLAS library is sufficient at link-time. */
void dgemv_(const char*, const BLASINT_TYPE*, const BLASINT_TYPE*,
  const double*, const double*, const BLASINT_TYPE*, const double*, const BLASINT_TYPE*,
  const double*, double*, const BLASINT_TYPE*);

void LIBXSMM_MATINIT(ITYPE, int seed, double* dst, BLASINT_TYPE nrows, BLASINT_TYPE ncols, BLASINT_TYPE ld, double scale);
void LIBXSMM_MATINIT(ITYPE, int seed, double* dst, BLASINT_TYPE nrows, BLASINT_TYPE ncols, BLASINT_TYPE ld, double scale)
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
  const BLASINT_TYPE lda = 3 < argc ? atoi(argv[3]) : m;
  const BLASINT_TYPE incx = 4 < argc ? atoi(argv[4]) : 1;
  const BLASINT_TYPE incy = 5 < argc ? atoi(argv[5]) : 1;
  const double alpha = 6 < argc ? atof(argv[6]) : 1.0;
  const double beta = 7 < argc ? atof(argv[7]) : 1.0;
  const char trans = 'N';
  double *a = 0, *x = 0, *y = 0;
  int i;

  if (8 < argc) size = atoi(argv[8]);
  a = (double*)malloc(lda * n * sizeof(double));
  x = (double*)malloc(incx * n * sizeof(double));
  y = (double*)malloc(incy * m * sizeof(double));
  printf("dgemv('%c', %i/*m*/, %i/*n*/,\n"
         "      %g/*alpha*/, %p/*a*/, %i/*lda*/,\n"
         "                  %p/*x*/, %i/*incx*/,\n"
         "       %g/*beta*/, %p/*y*/, %i/*incy*/)\n",
    trans, m, n, alpha, (const void*)a, lda,
                        (const void*)x, incx,
                  beta, (const void*)y, incy);

  assert(0 != a && 0 != x && 0 != y);
  LIBXSMM_MATINIT(ITYPE, 42, a, m, n, lda, 1.0);
  LIBXSMM_MATINIT(ITYPE, 24, x, n, 1, incx, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, y, m, 1, incy, 1.0);

  for (i = 0; i < size; ++i) {
    dgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
  printf("Called %i times.\n", size);

  free(a);
  free(x);
  free(y);

  return EXIT_SUCCESS;
}

