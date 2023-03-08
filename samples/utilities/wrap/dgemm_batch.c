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
#if defined(__INTEL_MKL__) && (110300 <= (10000*__INTEL_MKL__+100*__INTEL_MKL_MINOR__))
# define GEMM_BATCH dgemm_batch_
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

/** Function prototype for DGEMM_BATCH. */
void GEMM_BATCH(const char /*transa_array*/[], const char /*transb_array*/[],
  const BLASINT_TYPE /*m_array*/[], const BLASINT_TYPE /*n_array*/[], const BLASINT_TYPE /*k_array*/[],
  const double /*alpha_array*/[], const double* /*a_array*/[], const BLASINT_TYPE /*lda_array*/[],
                                  const double* /*b_array*/[], const BLASINT_TYPE /*ldb_array*/[],
  const double  /*beta_array*/[],       double* /*c_array*/[], const BLASINT_TYPE /*ldc_array*/[],
  const BLASINT_TYPE* /*group_count*/, const BLASINT_TYPE /*group_size*/[]);


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
  const BLASINT_TYPE batchsize = (7 < argc ? atoi(argv[7]) : 1000);
  const double alpha = (8 < argc ? atof(argv[8]) : (ALPHA));
  const double beta = (9 < argc ? atof(argv[9]) : (BETA));
  const char transa = 'N', transb = 'N';
  const BLASINT_TYPE na = lda * k, nb = ldb * n, nc = ldc * n;
  const BLASINT_TYPE group_count = 1;
  const double* *const pa = (const double**)malloc(sizeof(double*) * batchsize);
  const double* *const pb = (const double**)malloc(sizeof(double*) * batchsize);
  double* *const pc = (double**)malloc(sizeof(double*) * batchsize);
  double *const a = (double*)malloc(sizeof(double) * na * batchsize);
  double *const b = (double*)malloc(sizeof(double) * nb * batchsize);
  double *const c = (double*)malloc(sizeof(double) * nc * batchsize);
  const double scale = 1.0 / batchsize;
  int i = 0;

  assert(NULL != a && NULL != b && NULL != c && NULL != pa && NULL != pb && NULL != pc);
#if defined(GEMM_BATCH)
  if (10 < argc) nrepeat = atoi(argv[10]);
#else
  nrepeat = 0;
#endif

  printf(
    "dgemm_batch('%c', '%c', %i/*m*/, %i/*n*/, %i/*k*/,\n"
    "            %g/*alpha*/, %p/*a*/, %i/*lda*/,\n"
    "                        %p/*b*/, %i/*ldb*/,\n"
    "            %g/*beta*/,  %p/*c*/, %i/*ldc*/,\n"
    "            %i/*group_count*/, %i/*batchsize*/)\n",
    transa, transb, m, n, k, alpha, (const void*)a, lda,
                                    (const void*)b, ldb,
                              beta, (const void*)c, ldc,
                              group_count, batchsize);

#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < batchsize; ++i) { /* use pointers instead of indexes */
    pa[i] = a + i * na; pb[i] = b + i * nb; pc[i] = c + i * nc;
    init(42 + i, a + i * na, m, k, lda, scale);
    init(24 + i, b + i * nb, k, n, ldb, scale);
    if (0 != beta) { /* no need to initialize for beta=0 */
      init(0, c + i * nc, m, n, ldc, scale);
    }
  }

  { /* Call DGEMM_BATCH */
#if defined(GEMM_BATCH)
# if defined(_OPENMP)
    const double start = omp_get_wtime();
# endif
    for (i = 0; i < nrepeat; ++i) {
      GEMM_BATCH(&transa, &transb, &m, &n, &k,
        &alpha, pa, &lda, pb, &ldb,
        &beta,  pc, &ldc,
        &group_count, &batchsize);
    }
#endif
#if defined(GEMM_BATCH) && defined(_OPENMP)
    printf("Called %i times (%f s).\n", nrepeat, omp_get_wtime() - start);
#else
    printf("Called %i times.\n", nrepeat);
#endif
  }

  free(pa);
  free(pb);
  free(pc);
  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
