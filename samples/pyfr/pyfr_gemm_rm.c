/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
    //fprintf(stderr, "Invalid ./a,out M N K\n");
    exit(-1);
  }

  m = atoi(argv[1]);
  n = atoi(argv[2]);
  k = atoi(argv[3]);
  reps = atoi(argv[4]);
  // this is col-major what you want to use
  // for the sizes in question
  lda = k;
  ldb = n;
  ldc = n;

  if (n % nblock != 0) {
    fprintf(stderr, "N needs to be divisable by %i\n", nblock);
    exit(-1);
  }

  a  = (double*)_mm_malloc(lda*m*sizeof(double), 64);
  b  = (double*)_mm_malloc(ldb*k*sizeof(double), 64);
  c1 = (double*)_mm_malloc(ldc*m*sizeof(double), 64);
  c2 = (double*)_mm_malloc(ldc*m*sizeof(double), 64);

  #pragma omp parallel for
  for (i = 0; i < lda*m; i++) {
    a[i] = drand48();
  }

  #pragma omp parallel for
  for (i = 0; i < ldb*k; i++) {
    b[i] = drand48();
  }

  #pragma omp parallel for
  for (i = 0; i < ldc*m; i++) {
    c1[i] = 0;
    c2[i] = 0;
  }

  // JIT Kernel
  kernel = libxsmm_dmmdispatch(nblock, m, k, &ldb, &lda, &ldc, NULL, NULL, NULL, &l_prefetch_op );
  if (kernel == 0) {
    printf("JIT failed, exiting\n");
    exit(-1);
  }

  // init MKL
  dgemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c1, &ldc);

  #pragma omp parallel for
  for (i = 0; i < ldc*m; i++) {
    c1[i] = 0;
    c2[i] = 0;
  }

  gettimeofday(&l_start, NULL);
  for ( j = 0; j < reps; j++ ) {
    dgemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c1, &ldc);
  }
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);

  fprintf(stdout, "time[s] MKL     (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, l_total/(double)reps );
  fprintf(stdout, "GFLOPS  MKL     (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, (2.0 * (double)m * (double)n * (double)k * (double)reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    MKL     (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, ((double)sizeof(double) * (((double)m * (double)n) + ((double)k * (double)n)) * (double)reps * 1.0e-9) / l_total );

  gettimeofday(&l_start, NULL);
  for ( j = 0; j < reps; j++ ) {
    #pragma omp parallel for private(i)
    for ( i = 0; i < n; i+=nblock) {
      kernel( b+i, a, c2+i, NULL, NULL, NULL );
    }
    gettimeofday(&l_end, NULL);
  }
  l_total = sec(l_start, l_end);

  fprintf(stdout, "time[s] libxsmm (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, l_total/(double)reps );
  fprintf(stdout, "GFLOPS  libxsmm (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, (2.0 * (double)m * (double)n * (double)k * (double)reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    libxsmm (RM, M=%i, N=%i, K=%i): %f\n", m, n, k, ((double)sizeof(double) * (((double)m * (double)n) + ((double)k * (double)n)) * (double)reps * 1.0e-9) / l_total );

  // test result
  double max_error = 0.0;
  for ( i = 0; i < ldc*m; i++) {
    if (max_error < fabs(c1[i] - c2[i])) {
      max_error = fabs(c1[i] - c2[i]);
    }
  }
  printf("max error: %f\n\n", max_error);
}

