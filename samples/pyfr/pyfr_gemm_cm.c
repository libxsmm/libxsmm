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

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mkl.h>
#include <libxsmm.h>

static double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

int main(int argc, char *argv[])
{
  int n,m,k;
  int lda,ldb,ldc;
  double* a;
  double* b;
  double* c1;
  double* c2;
  struct timeval l_start, l_end;
  double l_total = 0.0;
  int reps, i, j;
  const int nblock = 16;
  double alpha = 1.0, beta = 1.0;
  char transa = 'N', transb = 'N';
  libxsmm_gemm_prefetch_type l_prefetch_op = LIBXSMM_PREFETCH_NONE;
  libxsmm_dmmfunction kernel = NULL;

  if (argc != 5) {
    fprintf(stderr, "Invalid ./a,out M N K reps\n");
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
    fprintf(stderr, "N needs to be divisable by %i\n", nblock);
    exit(-1);
  }

  a  = (double*)_mm_malloc(lda*k*sizeof(double), 64);
  b  = (double*)_mm_malloc(ldb*n*sizeof(double), 64);
  c1 = (double*)_mm_malloc(ldc*n*sizeof(double), 64);
  c2 = (double*)_mm_malloc(ldc*n*sizeof(double), 64);

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

  gettimeofday(&l_start, NULL);
  for ( j = 0; j < reps; j++ ) {
    dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c1, &ldc);
  }
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);

  fprintf(stdout, "time[s] MKL     (CM, M=%i, N=%i, K=%i): %f\n", m, n, k, l_total/(double)reps );
  fprintf(stdout, "GFLOPS  MKL     (CM, M=%i, N=%i, K=%i): %f\n", m, n, k, (2.0 * (double)m * (double)n * (double)k * (double)reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    MKL     (CM, M=%i, N=%i, K=%i): %f\n", m, n, k, ((double)sizeof(double) * (((double)m * (double)n) + ((double)k * (double)n)) * (double)reps * 1.0e-9) / l_total );

  gettimeofday(&l_start, NULL);
  for ( j = 0; j < reps; j++ ) {
    #pragma omp parallel for private(i)
    for ( i = 0; i < n; i+=nblock) {
      kernel( a, b+(ldb*i), c2+(ldc*i), NULL, NULL, NULL );
    }
    gettimeofday(&l_end, NULL);
  }
  l_total = sec(l_start, l_end);

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
}

