/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl.h>
#else /* prototypes for GEMM */
void dgemm_(const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
#endif


int main(int argc, char *argv[])
{
  int n,m,k;
  int lda,ldb,ldc;
  double* a;
  double* b;
  double* c1;
  double* c2;
  libxsmm_timer_tickint l_start, l_end;
  double l_total = 0.0;
  int reps, i, j;
  const int nblock = 16;
  double alpha = 1.0, beta = 1.0;
  char transa = 'N', transb = 'N';
  int l_prefetch_op = LIBXSMM_PREFETCH_NONE;
  libxsmm_dmmfunction kernel = NULL;

  if (argc != 5) {
    assert(0 < argc);
    fprintf(stderr, "Invalid: try %s M N K reps\n", argv[0]);
    exit(-1);
  }

  m = atoi(argv[1]);
  n = atoi(argv[2]);
  k = atoi(argv[3]);
  reps = atoi(argv[4]);
  /* this is col-major what you want to use for the sizes in question */
  lda = m;
  ldb = k;
  ldc = m;

  if (n % nblock != 0) {
    fprintf(stderr, "N needs to be divisible by %i\n", nblock);
    exit(-1);
  }

  a  = (double*)libxsmm_aligned_malloc(sizeof(double)*lda*k, 64);
  b  = (double*)libxsmm_aligned_malloc(sizeof(double)*ldb*n, 64);
  c1 = (double*)libxsmm_aligned_malloc(sizeof(double)*ldc*n, 64);
  c2 = (double*)libxsmm_aligned_malloc(sizeof(double)*ldc*n, 64);

  #pragma omp parallel for
  for (i = 0; i < lda*k; i++) {
    a[i] = libxsmm_rng_f64();
  }

  #pragma omp parallel for
  for (i = 0; i < ldb*n; i++) {
    b[i] = libxsmm_rng_f64();
  }

  #pragma omp parallel for
  for (i = 0; i < ldc*n; i++) {
    c1[i] = 0;
    c2[i] = 0;
  }

  /* JIT Kernel */
  kernel = libxsmm_dmmdispatch(m, nblock, k, NULL, NULL, NULL, NULL, NULL, NULL, &l_prefetch_op );

  /* init MKL */
  dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c1, &ldc);

  #pragma omp parallel for
  for (i = 0; i < ldc*n; i++) {
    c1[i] = 0;
    c2[i] = 0;
  }

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < reps; j++ ) {
    dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c1, &ldc);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);

  fprintf(stdout, "time[s] MKL     (CM, M=%i, N=%i, K=%i): %f\n", m, n, k, l_total/(double)reps );
  fprintf(stdout, "GFLOPS  MKL     (CM, M=%i, N=%i, K=%i): %f\n", m, n, k, (2.0 * (double)m * (double)n * (double)k * (double)reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    MKL     (CM, M=%i, N=%i, K=%i): %f\n", m, n, k, ((double)sizeof(double) * (((double)m * (double)n) + ((double)k * (double)n)) * (double)reps * 1.0e-9) / l_total );

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < reps; j++ ) {
    #pragma omp parallel for private(i)
    for ( i = 0; i < n; i+=nblock) {
      kernel( a, &b[ldb*i], &c2[ldc*i], NULL, NULL, NULL );
    }
    l_end = libxsmm_timer_tick();
  }
  l_total = libxsmm_timer_duration(l_start, l_end);

  fprintf(stdout, "time[s] libxsmm (CM, M=%i, N=%i, K=%i): %f\n", m, n, k, l_total/(double)reps );
  fprintf(stdout, "GFLOPS  libxsmm (CM, M=%i, N=%i, K=%i): %f\n", m, n, k, (2.0 * (double)m * (double)n * (double)k * (double)reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    libxsmm (CM, M=%i, N=%i, K=%i): %f\n", m, n, k, ((double)sizeof(double) * (((double)m * (double)n) + ((double)k * (double)n)) * (double)reps * 1.0e-9) / l_total );

  /* test result */
  double max_error = 0.0;
  for ( i = 0; i < ldc*n; i++) {
    if (max_error < fabs(c1[i] - c2[i])) {
      max_error = fabs(c1[i] - c2[i]);
    }
  }
  printf("max error: %f\n\n", max_error);

  return EXIT_SUCCESS;
}
