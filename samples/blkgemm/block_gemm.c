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

#include <libxsmm.h>
#include <libxsmm_timer.h>
#include <libxsmm_malloc.h>
#include <mkl.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef float real;

typedef struct libxsmm_blk_gemm_handle {
  int m;
  int n;
  int k;
  int bm;
  int bn;
  int bk;
} libxsmm_blk_gemm_handle;

void init_a( libxsmm_blk_gemm_handle* handle,
             real* libxsmm_mat_dst,
             real* colmaj_mat_src ) {
  int mb, kb, bm, bk;
#if defined(LIBXSMM_VLA)
  typedef float (*LIBXSMM_RESTRICT dst_type)[(handle->m)/(handle->bm)][handle->bk][handle->bm];
  typedef float (*LIBXSMM_RESTRICT src_type)[handle->m];
  const dst_type dst = (dst_type)libxsmm_mat_dst;
  const src_type src = (src_type)colmaj_mat_src;

  for ( kb = 0; kb < (handle->k)/(handle->bk); kb++ ) {
    for ( mb = 0; mb < (handle->m)/(handle->bm); mb++ ) {
      for ( bk = 0; bk < handle->bk; bk++ ) {
        for ( bm = 0; bm < handle->bm; bm++ ) {
          dst[kb][mb][bk][bm] = src[(kb*handle->bk)+bk][(mb*handle->bm)+bm];
        }
      }
    }
  }
#else
#error this code only works with LIBXSMM_VLA being available.
#endif
}

void init_b( libxsmm_blk_gemm_handle* handle,
             real* libxsmm_mat_dst,
             real* colmaj_mat_src ) {
  int kb, nb, bk, bn;
#if defined(LIBXSMM_VLA)
  typedef float (*LIBXSMM_RESTRICT dst_type)[(handle->k)/(handle->bk)][handle->bn][handle->bk];
  typedef float (*LIBXSMM_RESTRICT src_type)[handle->k];
  const dst_type dst = (dst_type)libxsmm_mat_dst;
  const src_type src = (src_type)colmaj_mat_src;

  for ( nb = 0; nb < (handle->n)/(handle->bn); nb++ ) {
    for ( kb = 0; kb < (handle->k)/(handle->bk); kb++ ) {
      for ( bn = 0; bn < handle->bn; bn++ ) {
        for ( bk = 0; bk < handle->bk; bk++ ) {
          dst[nb][kb][bn][bk] = src[(nb*handle->bn)+bn][(kb*handle->bk)+bk];
        }
      }
    }
  }
#else
#error this code only works with LIBXSMM_VLA being available.
#endif
}

void init_c( libxsmm_blk_gemm_handle* handle,
             real* libxsmm_mat_dst,
             real* colmaj_mat_src ) {
  int mb, nb, bm, bn;
#if defined(LIBXSMM_VLA)
  typedef float (*LIBXSMM_RESTRICT dst_type)[(handle->m)/(handle->bm)][handle->bn][handle->bm];
  typedef float (*LIBXSMM_RESTRICT src_type)[handle->m];
  const dst_type dst = (dst_type)libxsmm_mat_dst;
  const src_type src = (src_type)colmaj_mat_src;

  for ( nb = 0; nb < (handle->n)/(handle->bn); nb++ ) {
    for ( mb = 0; mb < (handle->m)/(handle->bm); mb++ ) {
      for ( bn = 0; bn < handle->bn; bn++ ) {
        for ( bm = 0; bm < handle->bm; bm++ ) {
          dst[nb][mb][bn][bm] = src[(nb*handle->bn)+bn][(mb*handle->bm)+bm];
        }
      }
    }
  }
#else
#error this code only works with LIBXSMM_VLA being available.
#endif
}

void compare_c( libxsmm_blk_gemm_handle* handle,
                real* libxsmm_mat_dst,
                real* colmaj_mat_src ) {
  int mb, nb, bm, bn;
  double max_error = 0.0;
  double src_norm = 0.0;
  double dst_norm = 0.0;
#if defined(LIBXSMM_VLA)
  typedef float (*LIBXSMM_RESTRICT dst_type)[(handle->m)/(handle->bm)][handle->bn][handle->bm];
  typedef float (*LIBXSMM_RESTRICT src_type)[handle->m];
  const dst_type dst = (dst_type)libxsmm_mat_dst;
  const src_type src = (src_type)colmaj_mat_src;

  for ( nb = 0; nb < (handle->n)/(handle->bn); nb++ ) {
    for ( mb = 0; mb < (handle->m)/(handle->bm); mb++ ) {
      for ( bn = 0; bn < handle->bn; bn++ ) {
        for ( bm = 0; bm < handle->bm; bm++ ) {
          double local_error = fabs((double)dst[nb][mb][bn][bm] - (double)src[(nb*handle->bn)+bn][(mb*handle->bm)+bm]);
          if (local_error > max_error) {
            max_error = local_error;
          }
          src_norm += (double)src[(nb*handle->bn)+bn][(mb*handle->bm)+bm];
          dst_norm += (double)dst[nb][mb][bn][bm];
        }
      }
    }
  }
#else
#error this code only works with LIBXSMM_VLA being available.
#endif
  printf(" max error: %f, sum BLAS: %f, sum LIBXSMM: %f \n", max_error, src_norm, dst_norm );
}

int main(int argc, char* argv []) {
  real *a, *b, *c, *a_gold, *b_gold, *c_gold;
  int M, N, K, LDA, LDB, LDC;
  real alpha, beta;
  unsigned long long start, end;
  double total, flops;
  int i, reps;
  size_t l;
  char trans;
  libxsmm_blk_gemm_handle handle;

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
    printf("\nUsage: ./block_gemm [M] [N] [K] [bm] [bn] [bk] [reps]\n\n");
    return 0;
  }

  /* setup defaults */
  handle.m = 2048;
  handle.n = 2048;
  handle.k = 2048;
  handle.bm = 32;
  handle.bn = 32;
  handle.bk = 32;
  reps = 100;

  /* reading new values from cli */
  i = 1;
  if (argc > i) handle.m      = atoi(argv[i++]);
  if (argc > i) handle.n      = atoi(argv[i++]);
  if (argc > i) handle.k      = atoi(argv[i++]);
  if (argc > i) handle.bm     = atoi(argv[i++]);
  if (argc > i) handle.bn     = atoi(argv[i++]);
  if (argc > i) handle.bk     = atoi(argv[i++]);
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
  printf(" Running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, reps=%i\n", M, N, K, handle.bm, handle.bn, handle.bk, reps );

  /* init random seed */
  srand48(1);

  /* allocate data */
  a      = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 2097152 );
  b      = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 2097152 );
  c      = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );
  a_gold = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 2097152 );
  b_gold = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 2097152 );
  c_gold = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );

  /* init data */
  for ( l = 0; l < (size_t)M * (size_t)K; l++ ) {
    a_gold[l] = (float)drand48();
  }
  for ( l = 0; l < (size_t)K * (size_t)N; l++ ) {
    b_gold[l] = (real)drand48();
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    c_gold[l] = (real)drand48();
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    c[l]      = (real)drand48();
  }
  init_a( &handle, a, a_gold );
  init_b( &handle, b, b_gold );

  /* check result */
  /* run LIBXSEMM, trans, alpha and beta are ignored */
#if 0
  libxsmm_blk_sgemm( &handle, &trans, &trans, &alpha, a, b, &beta, c );
#endif
  /* run BLAS */
  sgemm(&trans, &trans, &M, &N, &K, &alpha, a_gold, &LDA, b_gold, &LDB, &beta, c_gold, &LDC);
  /* compare result */
  init_c( &handle, c, c_gold );
  compare_c( &handle, c, c_gold );

  /* time BLAS */
  start = libxsmm_timer_tick();
  for ( i = 0; i < reps; i++ ) {
    sgemm(&trans, &trans, &M, &N, &K, &alpha, a_gold, &LDA, b_gold, &LDB, &beta, c_gold, &LDC);
  }
  end = libxsmm_timer_tick();
  total = libxsmm_timer_duration(start, end);
  printf("GFLOPS  (BLAS)    = %.5g\n", (flops*1e-9)/total);

  /* time libxsmm */
  start = libxsmm_timer_tick();
  for ( i = 0; i < reps; i++ ) {
#if 0
    libxsmm_blk_sgemm( &handle, &trans, &trans, &alpha, a, b, &beta, c );
#endif
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

