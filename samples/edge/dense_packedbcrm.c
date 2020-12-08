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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <libxsmm.h>

#if defined(__EDGE_EXECUTE_F32__)
#define REALTYPE float
#else
#define REALTYPE double
#endif


static void matMulFusedBC(       unsigned int    i_r,
                                 unsigned int    i_m,
                                 unsigned int    i_n,
                                 unsigned int    i_k,
                                 unsigned int    i_ldA,
                                 unsigned int    i_ldB,
                                 unsigned int    i_ldC,
                                 REALTYPE           i_beta,
                           const REALTYPE           *i_a,
                           const REALTYPE           *i_b,
                                 REALTYPE           *o_c ) {
  unsigned int l_m = 0;
  unsigned int l_n = 0;
  unsigned int l_r = 0;
  unsigned int l_k = 0;

  /* init result matrix */
  for ( l_m = 0; l_m < i_m; l_m++ ) {
    for ( l_n = 0; l_n < i_n; l_n++ ) {
      for ( l_r = 0; l_r < i_r; l_r++ ) {
        o_c[l_m*i_ldC*i_r + l_n*i_r + l_r] = (i_beta != (REALTYPE)0) ? o_c[l_m*i_ldC*i_r + l_n*i_r + l_r] * i_beta : 0;
      }
    }
  }
  /* perform matmul */
  for ( l_k = 0; l_k < i_k; l_k++ ) {
    for ( l_m = 0; l_m < i_m; l_m++ ) {
      for ( l_n = 0; l_n < i_n; l_n++ ) {
        for ( l_r = 0; l_r < i_r; l_r++ ) {
          o_c[l_m*i_ldC*i_r + l_n*i_r + l_r] += i_a[l_m*i_ldA + l_k] * i_b[l_k*i_ldB*i_r + l_n*i_r + l_r];
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
#if defined(__EDGE_EXECUTE_F32__)
  unsigned int l_r = 16;
#else
  unsigned int l_r = 8;
#endif
  unsigned int l_m = 1 < argc ? atoi(argv[1]) : 0;
  unsigned int l_n = 2 < argc ? atoi(argv[2]) : 0;
  unsigned int l_k = 3 < argc ? atoi(argv[3]) : 0;
  REALTYPE l_beta = (REALTYPE)(4 < argc ? atof(argv[4]) : 0);
  REALTYPE l_alpha = 1.0;
  unsigned int l_reps = 5 < argc ? atoi(argv[5]) : 0;
  double flops = 2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_r * (double)l_reps;

  REALTYPE* a = (REALTYPE*)  libxsmm_aligned_malloc( l_m*l_k*sizeof(REALTYPE), 64 );
  REALTYPE* b = (REALTYPE*)  libxsmm_aligned_malloc( l_k*l_n*l_r*sizeof(REALTYPE), 64 );
  REALTYPE* c1 = (REALTYPE*) libxsmm_aligned_malloc( l_m*l_n*l_r*sizeof(REALTYPE), 64 );
  REALTYPE* c2 = (REALTYPE*) libxsmm_aligned_malloc( l_m*l_n*l_r*sizeof(REALTYPE), 64 );

  libxsmm_descriptor_blob l_xgemm_blob;
  const libxsmm_gemm_descriptor* l_xgemm_desc = 0;
  LIBXSMM_MMFUNCTION_TYPE(REALTYPE) mykernel = NULL;
  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const int flags = LIBXSMM_GEMM_FLAGS('N', 'N');

  libxsmm_timer_tickint l_start, l_end;
  double l_total_ref, l_total_opt;
  double max_error = 0.0;
  double gflops_ref = 0.0;
  double gflops_opt = 0.0;
  double gflops_opt2 = 0.0;
  unsigned int i = 0;
  unsigned long long l_libxsmmflops;
  libxsmm_kernel_info l_kinfo;

  for ( i = 0; i < l_m*l_n*l_r; ++i ) {
    c1[i] = (REALTYPE)libxsmm_rng_f64();
  }
  for ( i = 0; i < l_m*l_n*l_r; ++i ) {
    c2[i] = c1[i];
  }
  for ( i = 0; i < l_m*l_k; ++i ) {
    a[i] = (REALTYPE)libxsmm_rng_f64();
  }
  for ( i = 0; i < l_k*l_n*l_r; ++i ) {
    b[i] = (REALTYPE)libxsmm_rng_f64();
  }

  /* JIT code */
  l_xgemm_desc = libxsmm_gemm_descriptor_dinit(&l_xgemm_blob, LIBXSMM_GEMM_PRECISION(REALTYPE),
    l_m, l_n, l_k, l_k, l_n, l_n, l_alpha, l_beta, flags, prefetch);
#if defined(__EDGE_EXECUTE_F32__)
  mykernel = libxsmm_create_packed_xgemm_bc_rm( l_xgemm_desc, l_r ).smm;
#else
  mykernel = libxsmm_create_packed_xgemm_bc_rm( l_xgemm_desc, l_r ).dmm;
#endif

  /* run reference */
  matMulFusedBC( l_r,
                  l_m, l_n, l_k,
                  l_k,
                  l_n,
                  l_n,
                  l_beta,
                  a,
                  b,
                  c1);

  /* run optimized */
  mykernel( a, b, c2 );

  /* check correctness */
  for ( i = 0; i < l_m*l_n*l_r; ++i ) {
    if ( max_error < fabs( c1[i] - c2[i] ) ) {
      max_error = fabs( c1[i] - c2[i] );
    }
  }

  printf("Max. Error: %f\n", max_error);

  /* lets run some performance test */
  l_start = libxsmm_timer_tick();
  for ( i = 0; i < l_reps; ++i ) {
    /* run reference */
    matMulFusedBC( l_r,
                    l_m, l_n, l_k,
                    l_k,
                    l_n,
                    l_n,
                    l_beta,
                    a,
                    b,
                    c1);
  }
  l_end = libxsmm_timer_tick();
  l_total_ref = libxsmm_timer_duration(l_start, l_end);

  l_start = libxsmm_timer_tick();
  for ( i = 0; i < l_reps; ++i ) {
    /* run optimized */
    mykernel( a, b, c2);
  }
  l_end = libxsmm_timer_tick();
  l_total_opt = libxsmm_timer_duration(l_start, l_end);
  libxsmm_get_kernel_info( LIBXSMM_CONST_VOID_PTR(mykernel), &l_kinfo);
  l_libxsmmflops = l_kinfo.nflops;

  gflops_ref = (flops/l_total_ref)/1e9;
  gflops_opt = (flops/l_total_opt)/1e9;
  gflops_opt2 = (((double)l_libxsmmflops*l_reps)/l_total_opt)/1e9;

  printf("GFLOPS ref: %f\n", gflops_ref);
  printf("GFLOPS opt, calculated: %f\n", gflops_opt);
  printf("GFLOPS opt, libxsmm:    %f\n", gflops_opt2);

  libxsmm_free( a );
  libxsmm_free( b );
  libxsmm_free( c1 );
  libxsmm_free( c2 );

  return 0;
}

