/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
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
void my_dgemm( const int* M, const int* N, const int* K, const double* alpha,
              const double* a, const int* LDA, const double* b, const int* LDB,
              const double* beta, double* c, const int* LDC ) {
  const int my_M = *M;
  const int my_N = *N;
  const int my_K = *K;
  const int my_LDA = *LDA;
  const int my_LDB = *LDB;
  const int my_LDC = *LDC;
  const double my_alpha = (double)*alpha;
  const double my_beta = (double)*beta;
  int m = 0, n = 0, k = 0;

  for ( n = 0; n < my_N; ++n ) {
    for ( m = 0; m < my_M; ++m ) {
      c[(n * my_LDC) + m] = my_beta * c[(n * my_LDC) + m];
      for ( k = 0; k < my_K; ++k ) {
        c[(n * my_LDC) + m] += my_alpha * a[(k * my_LDA) + m] * b[(n * my_LDB) + k];
      }
    }
  }
}
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
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
  char transa = 'N', transb = 'N';
#endif
  libxsmm_gemm_shape gemm_shape;
  const libxsmm_gemm_batch_reduce_config gemm_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_NONE, 0, 0, 0);
  const libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_gemmfunction kernel = NULL;
  libxsmm_gemm_param gemm_param;
  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );

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
  lda = k;
  ldb = n;
  ldc = n;

  if (n % nblock != 0) {
    fprintf(stderr, "N needs to be divisible by %i\n", nblock);
    exit(-1);
  }

  a  = (double*)libxsmm_aligned_malloc(sizeof(double)*lda*m, 64);
  b  = (double*)libxsmm_aligned_malloc(sizeof(double)*ldb*k, 64);
  c1 = (double*)libxsmm_aligned_malloc(sizeof(double)*ldc*m, 64);
  c2 = (double*)libxsmm_aligned_malloc(sizeof(double)*ldc*m, 64);

  #pragma omp parallel for
  for (i = 0; i < lda*m; i++) {
    a[i] = libxsmm_rng_f64();
  }

  #pragma omp parallel for
  for (i = 0; i < ldb*k; i++) {
    b[i] = libxsmm_rng_f64();
  }

  #pragma omp parallel for
  for (i = 0; i < ldc*m; i++) {
    c1[i] = 0;
    c2[i] = 0;
  }

  /* JIT Kernel */
  gemm_shape = libxsmm_create_gemm_shape(
    nblock, m, k, &ldb, &lda, &ldc, LIBXSMM_DATATYPE_F64,
    LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64 );
  kernel = libxsmm_dispatch_gemm_v2( gemm_shape, l_flags, l_prefetch_flags, gemm_brconfig );
  if (kernel == 0) {
    printf("JIT failed, exiting\n");
    exit(-1);
  }

  /* init MKL */
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
  dgemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c1, &ldc);
#else
  my_dgemm(&n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c1, &ldc);
#endif

  #pragma omp parallel for
  for (i = 0; i < ldc*m; i++) {
    c1[i] = 0;
    c2[i] = 0;
  }

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < reps; j++ ) {
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
    dgemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c1, &ldc);
#else
    my_dgemm(&n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c1, &ldc);
#endif
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);

  fprintf(stdout, "time[s] MKL     (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, l_total/(double)reps );
  fprintf(stdout, "GFLOPS  MKL     (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, (2.0 * (double)m * (double)n * (double)k * (double)reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    MKL     (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, ((double)sizeof(double) * (((double)m * (double)n) + ((double)k * (double)n)) * (double)reps * 1.0e-9) / l_total );

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < reps; j++ ) {
    #pragma omp parallel for private(i, gemm_param)
    for ( i = 0; i < n; i+=nblock) {
      gemm_param.a.primary = (void*)(b+i);
      gemm_param.b.primary = (void*)a;
      gemm_param.c.primary = (void*)(c2+i);
      kernel( &gemm_param );
    }
    l_end = libxsmm_timer_tick();
  }
  l_total = libxsmm_timer_duration(l_start, l_end);

  fprintf(stdout, "time[s] libxsmm (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, l_total/(double)reps );
  fprintf(stdout, "GFLOPS  libxsmm (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, (2.0 * (double)m * (double)n * (double)k * (double)reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    libxsmm (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, ((double)sizeof(double) * (((double)m * (double)n) + ((double)k * (double)n)) * (double)reps * 1.0e-9) / l_total );

  /* test result */
  double max_error = 0.0;
  for ( i = 0; i < ldc*m; i++) {
    if (max_error < fabs(c1[i] - c2[i])) {
      max_error = fabs(c1[i] - c2[i]);
    }
  }
  printf("max error: %f\n\n", max_error);

  return EXIT_SUCCESS;
}
