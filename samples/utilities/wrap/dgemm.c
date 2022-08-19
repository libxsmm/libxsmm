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

#if !defined(GEMM)
# define GEMM dgemm_
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
void GEMM(const char*, const char*, const BLASINT_TYPE*, const BLASINT_TYPE*, const BLASINT_TYPE*,
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
  const BLASINT_TYPE k = (3 < argc ? atoi(argv[3]) : m);
  const BLASINT_TYPE n = (2 < argc ? atoi(argv[2]) : k);
  const BLASINT_TYPE lda = (4 < argc ? atoi(argv[4]) : m);
  const BLASINT_TYPE ldb = (5 < argc ? atoi(argv[5]) : k);
  const BLASINT_TYPE ldc = (6 < argc ? atoi(argv[6]) : m);
  const double alpha = (7 < argc ? atof(argv[7]) : (ALPHA));
  const double beta = (8 < argc ? atof(argv[8]) : (BETA));
  const char transa = 'N', transb = 'N';
  const BLASINT_TYPE na = lda * k, nb = ldb * n, nc = ldc * n;
  double *const a = (double*)malloc(sizeof(double) * na);
  double *const b = (double*)malloc(sizeof(double) * nb);
  double *const c = (double*)malloc(sizeof(double) * nc);
  const double scale = 1.0;
  int i;

  assert(NULL != a && NULL != b && NULL != c);
  if (9 < argc) nrepeat = atoi(argv[9]);

  printf(
    "dgemm('%c', '%c', %i/*m*/, %i/*n*/, %i/*k*/,\n"
    "      %g/*alpha*/, %p/*a*/, %i/*lda*/,\n"
    "                  %p/*b*/, %i/*ldb*/,\n"
    "       %g/*beta*/, %p/*c*/, %i/*ldc*/)\n",
    transa, transb, m, n, k, alpha, (const void*)a, lda,
                                    (const void*)b, ldb,
                              beta, (const void*)c, ldc);

  init(42, a, m, k, lda, scale);
  init(24, b, k, n, ldb, scale);
  init( 0, c, m, n, ldc, scale);

  { /* Call DGEMM */
# if defined(_OPENMP)
    const double start = omp_get_wtime();
# endif
    for (i = 0; i < nrepeat; ++i) {
      GEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
# if defined(_OPENMP)
    printf("Called %i times (%f s).\n", nrepeat, omp_get_wtime() - start);
# else
    printf("Called %i times.\n", nrepeat);
# endif
  }

  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
