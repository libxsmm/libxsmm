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

#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && ( \
    (defined(__x86_64__) && 0 != (__x86_64__)) || \
    (defined(__amd64__) && 0 != (__amd64__)) || \
    (defined(_M_X64) || defined(_M_AMD64)) || \
    (defined(__i386__) && 0 != (__i386__)) || \
    (defined(_M_IX86)))
# include <mkl.h>
#endif
#if defined(__INTEL_MKL__) && (20200002 <= (10000*__INTEL_MKL__+100*__INTEL_MKL_MINOR__+__INTEL_MKL_UPDATE__))
# define GEMM_BATCH_STRIDED dgemm_batch_strided_
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

/** Function prototype for DGEMM_BATCH_STRIDED. */
void GEMM_BATCH_STRIDED(const char* /*transa*/, const char* /*transb*/,
  const BLASINT_TYPE* /*m*/, const BLASINT_TYPE* /*n*/, const BLASINT_TYPE* /*k*/,
  const double* /*alpha*/, const double* /*a*/, const BLASINT_TYPE* /*lda*/, const BLASINT_TYPE* /*stride_a*/,
                           const double* /*b*/, const BLASINT_TYPE* /*ldb*/, const BLASINT_TYPE* /*stride_b*/,
  const double* /*beta*/,        double* /*c*/, const BLASINT_TYPE* /*ldc*/, const BLASINT_TYPE* /*stride_c*/,
  const BLASINT_TYPE* /*batchsize*/);


void init(int seed, double* dst, BLASINT_TYPE nrows, BLASINT_TYPE ncols, BLASINT_TYPE ld, double scale);
void init(int seed, double* dst, BLASINT_TYPE nrows, BLASINT_TYPE ncols, BLASINT_TYPE ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  BLASINT_TYPE i = 0;
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
  const BLASINT_TYPE stride_a = (7 < argc ? atoi(argv[7]) : 0);
  const BLASINT_TYPE stride_b = (8 < argc ? atoi(argv[8]) : 0);
  const BLASINT_TYPE stride_c = (9 < argc ? atoi(argv[9]) : 0);
  const BLASINT_TYPE batchsize = (10 < argc ? atoi(argv[10]) : 1000);
  const double alpha = (11 < argc ? atof(argv[11]) : (ALPHA));
  const double beta = (12 < argc ? atof(argv[12]) : (BETA));
  const char transa = 'N', transb = 'N';
  const BLASINT_TYPE na = ((lda * k) < stride_a ? stride_a : (lda * k));
  const BLASINT_TYPE nb = ((ldb * n) < stride_b ? stride_b : (ldb * n));
  const BLASINT_TYPE nc = ((ldc * n) < stride_c ? stride_c : (ldc * n));
  double *const a = (double*)malloc(sizeof(double) * na * batchsize);
  double *const b = (double*)malloc(sizeof(double) * nb * batchsize);
  double *const c = (double*)malloc(sizeof(double) * nc * batchsize);
  const double scale = 1.0 / batchsize;
  int i = 0;

  assert(NULL != a && NULL != b && NULL != c);
#if defined(GEMM_BATCH_STRIDED)
  if (13 < argc) nrepeat = atoi(argv[13]);
#else
  nrepeat = 0;
#endif

  printf(
    "dgemm_batch_strided('%c', '%c', %i/*m*/, %i/*n*/, %i/*k*/,\n"
    "                    %g/*alpha*/, %p/*a*/, %i/*lda*/, %i/*stride_a*/,\n"
    "                                %p/*b*/, %i/*ldb*/, %i/*stride_b*/,\n"
    "                    %g/*beta*/,  %p/*c*/, %i/*ldc*/, %i/*stride_c*/,\n"
    "                    %i/*batchsize*/)\n",
    transa, transb, m, n, k, alpha, (const void*)a, lda, na,
                                    (const void*)b, ldb, nb,
                              beta, (const void*)c, ldc, nc,
                              batchsize);

#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < batchsize; ++i) { /* use pointers instead of indexes */
    init(42 + i, a + i * na, m, k, lda, scale);
    init(24 + i, b + i * nb, k, n, ldb, scale);
    if (0 != beta) { /* no need to initialize for beta=0 */
      init(0, c + i * nc, m, n, ldc, scale);
    }
  }

  { /* Call DGEMM_BATCH */
#if defined(GEMM_BATCH_STRIDED)
# if defined(_OPENMP)
    const double start = omp_get_wtime();
# endif
    for (i = 0; i < nrepeat; ++i) {
      GEMM_BATCH_STRIDED(&transa, &transb, &m, &n, &k,
        &alpha, a, &lda, &na, b, &ldb, &nb,
        &beta,  c, &ldc, &nc, &batchsize);
    }
#endif
#if defined(GEMM_BATCH_STRIDED) && defined(_OPENMP)
    printf("Called %i times (%f s).\n", nrepeat, omp_get_wtime() - start);
#else
    printf("Called %i times.\n", nrepeat);
#endif
  }

  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
