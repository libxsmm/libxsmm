/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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


void init(int seed, double* dst, BLASINT_TYPE nrows, BLASINT_TYPE ncols, BLASINT_TYPE ld, double scale);
void init(int seed, double* dst, BLASINT_TYPE nrows, BLASINT_TYPE ncols, BLASINT_TYPE ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  BLASINT_TYPE i = 0;
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
  init(42, a, m, n, lda, 1.0);
  init(24, x, n, 1, incx, 1.0);
  init( 0, y, m, 1, incy, 1.0);

  for (i = 0; i < size; ++i) {
    dgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
  printf("Called %i times.\n", size);

  free(a);
  free(x);
  free(y);

  return EXIT_SUCCESS;
}

