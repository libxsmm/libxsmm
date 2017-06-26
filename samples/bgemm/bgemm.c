/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/

#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif
#include <libxsmm_bgemm.h>

#if defined(_WIN32) || defined(__CYGWIN__)
/* note: this does not reproduce 48-bit RNG quality */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif


int main(int argc, char* argv[])
{
  real *a = 0, *b = 0, *c = 0, *a_gold = 0, *b_gold = 0, *c_gold = 0;
  int M = 2048, N = 2048, K = 2048, LDA = 0, LDB = 0, LDC = 0;
  int bm = 32, bn = 32, bk = 32;
  int order = 0, reps = 100;
  char transa = 'N', transb = 'N';
  real alpha = 1, beta = 1;
  unsigned long long start = 0;
  double total = 0, flops = 0;
  size_t li = 0; /* linear index */
  int i;

  libxsmm_malloc_info info_a, info_b, info_b;
  libxsmm_bgemm_handle* handle = 0;

  /* check command line */
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./block_gemm [M] [N] [K] [bm] [bn] [bk] [order] [reps]\n\n");
    return 0;
  }

  /* reading values from CLI */
  i = 1;
  if (argc > i) M     = atoi(argv[i++]);
  if (argc > i) N     = atoi(argv[i++]);
  if (argc > i) K     = atoi(argv[i++]);
  if (argc > i) bm    = atoi(argv[i++]);
  if (argc > i) bn    = atoi(argv[i++]);
  if (argc > i) bk    = atoi(argv[i++]);
  if (argc > i) order = atoi(argv[i++]);
  if (argc > i) reps  = atoi(argv[i++]);

  LDA = M; LDB = K; LDC = M;
  flops = 2.0 * M * N * K * reps;

  /* allocate data */
  a = (real*)libxsmm_aligned_malloc(M * K * sizeof(real), 2097152);
  b = (real*)libxsmm_aligned_malloc(K * N * sizeof(real), 2097152);
  c = (real*)libxsmm_aligned_malloc(M * N * sizeof(real), 2097152);
  libxsmm_get_malloc_info(a, &info_a);
  libxsmm_get_malloc_info(b, &info_b);
  libxsmm_get_malloc_info(c, &info_c);

  /* init random seed and print some info */
  printf(" running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, reps=%i\n", M, N, K, bm, bn, bk, reps);
  printf(" working set size: A: %f, B: %f, C: %f, Total: %f in MiB\n",
    (double)info_a.size / (1024.0 * 1024.0), (double)info_b.size / (1024.0 * 1024.0), (double)info_c.size / (1024.0 * 1024.0),
    (double)(info_a.size + info_b.size + info_c.size) / (1024.0 * 1024.0));

  /* allocate Gold data */
  a_gold = (real*)libxsmm_aligned_malloc(M * K * sizeof(real), 2097152);
  b_gold = (real*)libxsmm_aligned_malloc(K * N * sizeof(real), 2097152);
  c_gold = (real*)libxsmm_aligned_malloc(M * N * sizeof(real), 2097152);

  /* init data */
  srand48(1);
  for ( li = 0; li < (size_t)M * (size_t)K; li++ ) {
    a_gold[li] = (real)drand48();
  }
  for ( li = 0; li < (size_t)K * (size_t)N; li++ ) {
    b_gold[li] = (real)drand48();
  }
  for ( li = 0; li < (size_t)M * (size_t)N; li++ ) {
    c_gold[li] = (real)0.0;
  }
  for ( li = 0; li < (size_t)M * (size_t)N; li++ ) {
    c[li] = (real)0.0;
  }

  handle = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION_F32, order, M, N, K, bm, bn, bk);
  libxsmm_bgemm_init_a(&handle, a, a_gold);
  libxsmm_bgemm_init_b(&handle, b, b_gold);

  handle._l_kernel =_KERNEL_JIT(bm, bn, bk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
#ifdef _USE_LIBXSMM_PREFETCH
  libxsmm_prefetch_type l_prefetch_op = LIBXSMM_PREFETCH_AL2BL2_VIA_C;
  handle._l_kernel_pf =_KERNEL_JIT(bm, bn, bn, NULL, NULL, NULL, NULL, NULL, NULL, &l_prefetch_op );
#endif

  /* check result */
  /* run LIBXSEMM, trans, alpha and beta are ignored */
  libxsmm_bgemm_omp(&handle, transa, transb, &alpha, a, b, &beta, c);
  /* run BLAS */
  LIBXSMM_FSYMBOL(sgemm)(&transa, &transb, &M, &N, &K, &alpha, a_gold, &LDA, b_gold, &LDB, &beta, c_gold, &LDC);
  /* compare result */
  libxsmm_bgemm_check_c(&handle, c, c_gold);

  /* time BLAS */
  start = libxsmm_timer_tick();
  for ( i = 0; i < reps; i++ ) {
    LIBXSMM_FSYMBOL(sgemm)(&transa, &transb, &M, &N, &K, &alpha, a_gold, &LDA, b_gold, &LDB, &beta, c_gold, &LDC);
  }
  total = libxsmm_timer_duration(start, libxsmm_timer_tick());
  printf("GFLOPS  (BLAS)    = %.5g\n", (flops * 1-9) / total);

  /* measure execution time */
  start = libxsmm_timer_tick();
  for ( i = 0; i < reps; i++ ) {
    libxsmm_bgemm_omp( &handle, transa, transb, &alpha, a, b, &beta, c );
  }
  total = libxsmm_timer_duration(start, libxsmm_timer_tick());
  printf("GFLOPS  (LIBXSMM) = %.5g\n", (flops * 1e-9) / total);

  /* free data */
  libxsmm_bgemm_handle_destroy(&handle);
  libxsmm_free(a_gold);
  libxsmm_free(b_gold);
  libxsmm_free(c_gold);
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);

  return 0;
}

