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
#include "libxsmm_blkgemm.h"

#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
/* note: this does not reproduce 48-bit RNG quality */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

int main(int argc, char* argv []) {
  real *a, *b, *c, *a_gold, *b_gold, *c_gold;
  int M, N, K, LDA, LDB, LDC;
  real alpha, beta;
  unsigned long long start, end;
  double total, flops;
  int i, reps;
  size_t l;
  char trans;
  libxsmm_blkgemm_handle handle;

  /* init */
/*
  a = 0;
  b = 0;
  c = 0;
  a_gold = 0;
  b_gold = 0;
  c_gold = 0;
*/
  M = 0;
  N = 0;
  K = 0;
  LDA = 0;
  LDB = 0;
  LDC = 0;
  alpha = (real)1.0;
  beta = (real)1.0;
  start = 0;
  end = 0;
  total = 0.0;
  flops = 0.0;
  i = 0;
  l = 0;
  reps = 0;
  trans = 'N';

  /* check command line */
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./block_gemm [M] [N] [K] [bm] [bn] [bk] [b_m1] [b_n1] [b_k1] [b_k2] [reps]\n\n");
    return 0;
  }

  /* setup defaults */
  handle.m = 2048;
  handle.n = 2048;
  handle.k = 2048;
  handle._ORDER = 0;
  handle.bm = 32;
  handle.bn = 32;
  handle.bk = 32;
  handle.b_m1 = 1;
  handle.b_n1 = 1;
  handle.b_k1 = 1;
  handle.b_k2 = 1;
  reps = 100;

  /* reading new values from cli */
  i = 1;
  if (argc > i) reps          = atoi(argv[i++]);
  if (argc > i) handle.m      = atoi(argv[i++]);
  if (argc > i) handle.n      = atoi(argv[i++]);
  if (argc > i) handle.k      = atoi(argv[i++]);
  if (argc > i) handle._ORDER = atoi(argv[i++]);
  if (argc > i) handle.bm     = atoi(argv[i++]);
  if (argc > i) handle.bn     = atoi(argv[i++]);
  if (argc > i) handle.bk     = atoi(argv[i++]);
  if (argc > i) handle.b_m1   = atoi(argv[i++]);
  if (argc > i) handle.b_n1   = atoi(argv[i++]);
  if (argc > i) handle.b_k1   = atoi(argv[i++]);
  if (argc > i) handle.b_k2   = atoi(argv[i++]);
  if (argc > i) reps          = atoi(argv[i++]);
  M = handle.m;
  LDA = handle.m;
  N = handle.n;
  LDB = handle.k;
  K = handle.k;
  LDC = handle.m;
  alpha = (real)1.0;
  beta = (real)1.0;
  flops = (double)M * (double)N * (double)K * (double)2.0 * (double)reps;

  /* check for valid blocking and JIT-kernel */
  if ( handle.m % handle.bm != 0 ) {
    printf( " M needs to be a multiple of bm... exiting!\n" );
    return -1;
  }
  if ( handle.n % handle.bn != 0 ) {
    printf( " N needs to be a multiple of bn... exiting!\n" );
    return -2;
  }
  if ( handle.k % handle.bk != 0 ) {
    printf( " K needs to be a multiple of bk... exiting!\n" );
    return -3;
  }
  if ( handle.m % handle.b_m1 != 0 ) {
    printf( " M needs to be a multiple of b_m1... exiting!\n" );
    return -4;
  }
  if ( handle.n % handle.b_n1 != 0 ) {
    printf( " N needs to be a multiple of b_n1... exiting!\n" );
    return -5;
  }
  if ( handle.k % handle.b_k1 != 0 ) {
    printf( " K needs to be a multiple of b_k1... exiting!\n" );
    return -6;
  }
  if ( handle.m/handle.b_m1 % handle.bm != 0 ) {
    printf( " m/b_m1 needs to be a multiple of bm... exiting!\n" );
    return -7;
  }
  if ( handle.n/handle.b_n1 % handle.bn != 0 ) {
    printf( " n/b_n1 needs to be a multiple of bn... exiting!\n" );
    return -8;
  }
  if ( handle.k/handle.b_k1/handle.b_k2 % handle.bk != 0 ) {
    printf( " k/b_k1/b_k2 needs to be a multiple of bk... exiting!\n" );
    return -9;
  }
  handle.mb = handle.m / handle.bm;
  handle.nb = handle.n / handle.bn;
  handle.kb = handle.k / handle.bk;

  /* init random seed and print some info */
  printf(" running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, b_m1=%i, b_n1=%i, b_k1=%i, b_k2=%i, reps=%i\n", M, N, K, handle.bm, handle.bn, handle.bk, handle.b_m1, handle.b_n1, handle.b_k1, handle.b_k2, reps );
  printf(" working set size: A: %f, B: %f, C: %f, Total: %f in MiB\n", ((double)(M*K*sizeof(real)))/(1024.0*1024.0),
                                                                       ((double)(K*N*sizeof(real)))/(1024.0*1024.0),
                                                                       ((double)(M*N*sizeof(real)))/(1024.0*1024.0),
                                                                       ((double)(M*N*sizeof(real)+M*K*sizeof(real)+N*K*sizeof(real)))/(1024.0*1024.0) );
  srand48(1);

#if defined(MKL_ENABLE_AVX512) /* AVX-512 instruction support */
  mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif

  /* allocate data */
  a      = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 2097152 );
  b      = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 2097152 );
  c      = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );
  a_gold = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 2097152 );
  b_gold = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 2097152 );
  c_gold = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );

  /* init data */
  for ( l = 0; l < (size_t)M * (size_t)K; l++ ) {
    a_gold[l] = (real)drand48();
  }
  for ( l = 0; l < (size_t)K * (size_t)N; l++ ) {
    b_gold[l] = (real)drand48();
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    c_gold[l] = (real)0.0;
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    c[l]      = (real)0.0;
  }
  libxsmm_blksgemm_init_a( &handle, a, a_gold );
  libxsmm_blksgemm_init_b( &handle, b, b_gold );

  handle._ORDER = 0;
  handle.C_pre_init = 0;
  handle._wlock = NULL; 
  handle.bar = NULL;
  
  libxsmm_blkgemm_handle_alloc(&handle, handle.m, handle.bm, handle.n, handle.bn);

  handle._l_kernel =_KERNEL_JIT(handle.bm, handle.bn, handle.bk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
#ifdef _USE_LIBXSMM_PREFETCH
  libxsmm_prefetch_type l_prefetch_op = LIBXSMM_PREFETCH_AL2BL2_VIA_C;
  handle._l_kernel_pf =_KERNEL_JIT(handle.bm, handle.bn, handle.bn, NULL, NULL, NULL, NULL, NULL, NULL, &l_prefetch_op );
#endif

  /* check result */
  /* run LIBXSEMM, trans, alpha and beta are ignored */
  libxsmm_blksgemm_exec( &handle, trans, trans, &alpha, a, b, &beta, c );
  /* run BLAS */
  LIBXSMM_FSYMBOL(sgemm)(&trans, &trans, &M, &N, &K, &alpha, a_gold, &LDA, b_gold, &LDB, &beta, c_gold, &LDC);
  /* compare result */
  libxsmm_blksgemm_check_c( &handle, c, c_gold );

  /* time BLAS */
  start = libxsmm_timer_tick();
  for ( i = 0; i < reps; i++ ) {
    LIBXSMM_FSYMBOL(sgemm)(&trans, &trans, &M, &N, &K, &alpha, a_gold, &LDA, b_gold, &LDB, &beta, c_gold, &LDC);
  }
  end = libxsmm_timer_tick();
  total = libxsmm_timer_duration(start, end);
  printf("GFLOPS  (BLAS)    = %.5g\n", (flops*1e-9)/total);

  /* time libxsmm */
  start = libxsmm_timer_tick();
  for ( i = 0; i < reps; i++ ) {
    libxsmm_blksgemm_exec( &handle, trans, trans, &alpha, a, b, &beta, c );
  }
  end = libxsmm_timer_tick();
  total = libxsmm_timer_duration(start, end);
  printf("GFLOPS  (LIBXSMM) = %.5g\n", (flops*1e-9)/total);

  /* free data */
  libxsmm_free( a );
  libxsmm_free( b );
  libxsmm_free( c );
  libxsmm_free( a_gold );
  libxsmm_free( b_gold );
  libxsmm_free( c_gold );

  return 0;
}

