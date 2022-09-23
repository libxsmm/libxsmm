/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(GEMV)
# define GEMV dgemv_
#endif
#if !defined(BLASINT_TYPE)
# define BLASINT_TYPE int
#endif
#if !defined(ALPHA)
# define ALPHA 1
#endif
#if !defined(BETA)
# define BETA 1
#endif

/** Function prototype for DGEMM; this way any kind of LAPACK/BLAS library is sufficient at link-time. */
void GEMV(const char*, const BLASINT_TYPE*, const BLASINT_TYPE*,
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
  int nrepeat = (2 == argc ? atoi(argv[1]) : 500);
  const BLASINT_TYPE m = (2 < argc ? atoi(argv[1]) : 23);
  const BLASINT_TYPE n = (2 < argc ? atoi(argv[2]) : m);
  const BLASINT_TYPE lda = (3 < argc ? atoi(argv[3]) : m);
  const BLASINT_TYPE incx = (4 < argc ? atoi(argv[4]) : 1);
  const BLASINT_TYPE incy = (5 < argc ? atoi(argv[5]) : 1);
  const double alpha = (6 < argc ? atof(argv[6]) : (ALPHA));
  const double beta = (7 < argc ? atof(argv[7]) : (BETA));
  const char trans = 'N';
  const BLASINT_TYPE na = lda * n, nx = incx * n, ny = incy * m;
  double *const a = (double*)malloc(sizeof(double) * na);
  double *const x = (double*)malloc(sizeof(double) * nx);
  double *const y = (double*)malloc(sizeof(double) * ny);
  const double scale = 1.0;
  int i;

  assert(NULL != a && NULL != x && NULL != y);
  if (8 < argc) nrepeat = atoi(argv[8]);

  printf(
    "dgemv('%c', %i/*m*/, %i/*n*/,\n"
    "      %g/*alpha*/, %p/*a*/, %i/*lda*/,\n"
    "                  %p/*x*/, %i/*incx*/,\n"
    "       %g/*beta*/, %p/*y*/, %i/*incy*/)\n",
    trans, m, n, alpha, (const void*)a, lda,
                        (const void*)x, incx,
                  beta, (const void*)y, incy);

  init(42, a, m, n, lda, scale);
  init(24, x, n, 1, incx, scale);
  init( 0, y, m, 1, incy, scale);

  { /* Call DGEMM */
# if defined(_OPENMP)
    const double start = omp_get_wtime();
# endif
    for (i = 0; i < nrepeat; ++i) {
      GEMV(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
# if defined(_OPENMP)
    printf("Called %i times (%f s).\n", nrepeat, omp_get_wtime() - start);
# else
    printf("Called %i times.\n", nrepeat);
# endif
  }

  free(a);
  free(x);
  free(y);

  return EXIT_SUCCESS;
}
